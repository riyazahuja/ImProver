lemma regularTopology.isLocallySurjective_iff [Preregular C] {F G : Cᵒᵖ ⥤ D} (f : F ⟶ G) :
    Presheaf.IsLocallySurjective (regularTopology C) f ↔
      ∀ (X : C) (y : G.obj ⟨X⟩), (∃ (X' : C) (φ : X' ⟶ X) (_ : EffectiveEpi φ) (x : F.obj ⟨X'⟩),
        f.app ⟨X'⟩ x = G.map ⟨φ⟩ y) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.ConcreteCategory D
    inst✝ : CategoryTheory.Preregular C
    F G : CategoryTheory.Functor (Opposite C) D
    f : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.regularTopo …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.regularTopology  …
    -/
  · intro ⟨h⟩ X y
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      h : ∀ {U : C} (s : (CategoryTheory.forget D).obj (G.obj { unop := U })), Membe …
      X : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := X })
      ⊢ Exists fun X' => Exists fun φ => Exists fun x => Exists fun x => Eq ((f.app  …
    -/
    specialize h y
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      X : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := X })
      h : Membership.mem ((CategoryTheory.regularTopology C) X) (CategoryTheory.Pres …
      ⊢ Exists fun X' => Exists fun φ => Exists fun x => Exists fun x => Eq ((f.app  …
    -/
    rw [regularTopology.mem_sieves_iff_hasEffectiveEpi] at h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      X : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := X })
      h : Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) ((Cate …
      ⊢ Exists fun X' => Exists fun φ => Exists fun x => Exists fun x => Eq ((f.app  …
    -/
    obtain ⟨X', π, h, h'⟩ := h
    /-
      case mp.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      X : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := X })
      X' : C
      π : Quiver.Hom X' X
      h : CategoryTheory.EffectiveEpi π
      h' : (CategoryTheory.Presheaf.imageSieve f y).arrows π
      ⊢ Exists fun X' => Exists fun φ => Exists fun x => Exists fun x => Eq ((f.app  …
    -/
    exact ⟨X', π, h, h'⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      ⊢ (∀ (X : C) (y : (CategoryTheory.forget D).obj (G.obj { unop := X })), Exists …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      h : ∀ (X : C) (y : (CategoryTheory.forget D).obj (G.obj { unop := X })), Exist …
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.regularTopology  …
    -/
    refine ⟨fun y ↦ ?_⟩
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      h : ∀ (X : C) (y : (CategoryTheory.forget D).obj (G.obj { unop := X })), Exist …
      U✝ : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := U✝ })
      ⊢ Membership.mem ((CategoryTheory.regularTopology C) U✝) (CategoryTheory.Presh …
    -/
    obtain ⟨X', π, h, h'⟩ := h _ y
    /-
      case mpr.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      h✝ : ∀ (X : C) (y : (CategoryTheory.forget D).obj (G.obj { unop := X })), Exis …
      U✝ : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := U✝ })
      X' : C
      π : Quiver.Hom X' U✝
      h : CategoryTheory.EffectiveEpi π
      h' : Exists fun x => Eq ((f.app { unop := X' }) x) ((G.map { unop := π }) y)
      ⊢ Membership.mem ((CategoryTheory.regularTopology C) U✝) (CategoryTheory.Presh …
    -/
    rw [regularTopology.mem_sieves_iff_hasEffectiveEpi]
    /-
      case mpr.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.ConcreteCategory D
      inst✝ : CategoryTheory.Preregular C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      h✝ : ∀ (X : C) (y : (CategoryTheory.forget D).obj (G.obj { unop := X })), Exis …
      U✝ : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := U✝ })
      X' : C
      π : Quiver.Hom X' U✝
      h : CategoryTheory.EffectiveEpi π
      h' : Exists fun x => Eq ((f.app { unop := X' }) x) ((G.map { unop := π }) y)
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) ((Catego …
    -/
    exact ⟨X', π, h, h'⟩
    /-
      🎉 no goals
    -/


lemma extensiveTopology.surjective_of_isLocallySurjective_sheaf_of_types [FinitaryPreExtensive C]
    {F G : Cᵒᵖ ⥤ Type w} (f : F ⟶ G) [PreservesFiniteProducts F] [PreservesFiniteProducts G]
      (h : Presheaf.IsLocallySurjective (extensiveTopology C) f) {X : C} :
        Function.Surjective (f.app (op X)) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    h : CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.extensiveTopol …
    X : C
    ⊢ Function.Surjective (f.app { unop := X })
  -/
  intro x
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    h : CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.extensiveTopol …
    X : C
    x : G.obj { unop := X }
    ⊢ Exists fun a => Eq (f.app { unop := X } a) x
  -/
  replace h := h.1 x
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    h : Membership.mem ((CategoryTheory.extensiveTopology C) X) (CategoryTheory.Pr …
    ⊢ Exists fun a => Eq (f.app { unop := X } a) x
  -/
  rw [mem_sieves_iff_contains_colimit_cofan] at h
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    h : Exists fun α => Exists fun x_1 => Exists fun Y => Exists fun π => And (Non …
    ⊢ Exists fun a => Eq (f.app { unop := X } a) x
  -/
  obtain ⟨α, _, Y, π, h, h'⟩ := h
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    ⊢ Exists fun a => Eq (f.app { unop := X } a) x
  -/
  let y : (a : α) → (F.obj ⟨Y a⟩) := fun a ↦ (h' a).choose
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    ⊢ Exists fun a => Eq (f.app { unop := X } a) x
  -/
  let _ : Fintype α := Fintype.ofFinite _
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ⊢ Exists fun a => Eq (f.app { unop := X } a) x
  -/
  let ht := (Types.productLimitCone (fun a ↦ F.obj ⟨Y a⟩)).isLimit
  let ht' := (Functor.Initial.isLimitWhiskerEquiv (Discrete.opposite α).inverse
    (Cocone.op (Cofan.mk X π))).symm h.some.op
  let i : ((a : α) → (F.obj ⟨Y a⟩)) ≅ (F.obj ⟨X⟩) :=
    ht.conePointsIsoOfNatIso (isLimitOfPreserves F ht')
      (Discrete.natIso (fun _ ↦ (Iso.refl (F.obj ⟨_⟩))))
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    ⊢ Exists fun a => Eq (f.app { unop := X } a) x
  -/
  refine ⟨i.hom y, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    ⊢ Eq (f.app { unop := X } (i.hom y)) x
  -/
  apply Concrete.isLimit_ext _ (isLimitOfPreserves G ht')
  /-
    case intro.intro.intro.intro.intro.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    ⊢ ∀ (j : CategoryTheory.Discrete α), Eq (((G.mapCone (CategoryTheory.Limits.Co …
  -/
  intro ⟨a⟩
  simp only [Functor.comp_obj, Discrete.opposite_inverse_obj, Functor.op_obj, Discrete.functor_obj,
    Functor.mapCone_pt, Cone.whisker_pt, Cocone.op_pt, Cofan.mk_pt, Functor.const_obj_obj,
    Functor.mapCone_π_app, Cone.whisker_π, Cocone.op_π, whiskerLeft_app, NatTrans.op_app,
    Cofan.mk_ι_app]
  /-
    case intro.intro.intro.intro.intro.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    a : α
    ⊢ Eq ((G.map (π a).op) (f.app { unop := X } (i.hom y))) ((G.map (π a).op) x)
  -/
  have : f.app ⟨Y a⟩ (y a) = G.map (π a).op x := (h' a).choose_spec
  /-
    case intro.intro.intro.intro.intro.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    a : α
    this : Eq (f.app { unop := Y a } (y a)) (G.map (π a).op x)
    ⊢ Eq ((G.map (π a).op) (f.app { unop := X } (i.hom y))) ((G.map (π a).op) x)
  -/
  change _ = G.map (π a).op x
  /-
    case intro.intro.intro.intro.intro.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    a : α
    this : Eq (f.app { unop := Y a } (y a)) (G.map (π a).op x)
    ⊢ Eq ((G.map (π a).op) (f.app { unop := X } (i.hom y))) (G.map (π a).op x)
  -/
  rw [← this]
  /-
    case intro.intro.intro.intro.intro.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    a : α
    this : Eq (f.app { unop := Y a } (y a)) (G.map (π a).op x)
    ⊢ Eq ((G.map (π a).op) (f.app { unop := X } (i.hom y))) (f.app { unop := Y a } …
  -/
  erw [← NatTrans.naturality_apply (φ := f)]
  /-
    case intro.intro.intro.intro.intro.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    a : α
    this : Eq (f.app { unop := Y a } (y a)) (G.map (π a).op x)
    ⊢ Eq ((f.app { unop := Y a }) ((F.map (π a).op) (i.hom y))) (f.app { unop := Y …
  -/
  change f.app _ ((i.hom ≫ F.map (π a).op) y) = _
  /-
    case intro.intro.intro.intro.intro.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    a : α
    this : Eq (f.app { unop := Y a } (y a)) (G.map (π a).op x)
    ⊢ Eq (f.app { unop := Y a } (CategoryTheory.CategoryStruct.comp i.hom (F.map ( …
  -/
  erw [IsLimit.map_π]
  /-
    case intro.intro.intro.intro.intro.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) (Type w)
    f : Quiver.Hom F G
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
    X : C
    x : G.obj { unop := X }
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
    h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f x).arrows (π a)
    y : (a : α) → F.obj { unop := Y a } := fun a => Exists.choose ⋯
    x✝ : Fintype α := Fintype.ofFinite α
    ht : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Types.productLimitCo …
    ht' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Cone.whisker (Categ …
    i : CategoryTheory.Iso ((a : α) → F.obj { unop := Y a }) (F.obj { unop := X }) …
    a : α
    this : Eq (f.app { unop := Y a } (y a)) (G.map (π a).op x)
    ⊢ Eq (f.app { unop := Y a } (CategoryTheory.CategoryStruct.comp ((CategoryTheo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-26")]
alias extensiveTopology.surjective_of_isLocallySurjective_sheafOfTypes :=
  extensiveTopology.surjective_of_isLocallySurjective_sheaf_of_types


lemma extensiveTopology.presheafIsLocallySurjective_iff [FinitaryPreExtensive C] {F G : Cᵒᵖ ⥤ D}
    (f : F ⟶ G) [PreservesFiniteProducts F] [PreservesFiniteProducts G]
      [PreservesFiniteProducts (forget D)] : Presheaf.IsLocallySurjective (extensiveTopology C) f ↔
        ∀ (X : C), Function.Surjective (f.app (op X)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.FinitaryPreExtensive C
    F G : CategoryTheory.Functor (Opposite C) D
    f : Quiver.Hom F G
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.extensiveTo …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.extensiveTopolog …
    -/
  · rw [Presheaf.isLocallySurjective_iff_whisker_forget (J := extensiveTopology C)]
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.extensiveTopolog …
    -/
    exact fun h _ ↦ surjective_of_isLocallySurjective_sheaf_of_types (whiskerRight f (forget D)) h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      ⊢ (∀ (X : C), Function.Surjective ⇑(f.app { unop := X })) → CategoryTheory.Pre …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      h : ∀ (X : C), Function.Surjective ⇑(f.app { unop := X })
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.extensiveTopolog …
    -/
    refine ⟨fun {X} y ↦ ?_⟩
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      h : ∀ (X : C), Function.Surjective ⇑(f.app { unop := X })
      X : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := X })
      ⊢ Membership.mem ((CategoryTheory.extensiveTopology C) X) (CategoryTheory.Pres …
    -/
    obtain ⟨x, hx⟩ := h X y
    /-
      case mpr.intro
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      h : ∀ (X : C), Function.Surjective ⇑(f.app { unop := X })
      X : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := X })
      x : (CategoryTheory.forget D).obj (F.obj { unop := X })
      hx : Eq ((f.app { unop := X }) x) y
      ⊢ Membership.mem ((CategoryTheory.extensiveTopology C) X) (CategoryTheory.Pres …
    -/
    convert (extensiveTopology C).top_mem' X
    /-
      case h.e'_5
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      h : ∀ (X : C), Function.Surjective ⇑(f.app { unop := X })
      X : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := X })
      x : (CategoryTheory.forget D).obj (F.obj { unop := X })
      hx : Eq ((f.app { unop := X }) x) y
      ⊢ Eq (CategoryTheory.Presheaf.imageSieve f y) Top.top
    -/
    rw [← Sieve.id_mem_iff_eq_top]
    /-
      case h.e'_5
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁴ : CategoryTheory.ConcreteCategory D
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      h : ∀ (X : C), Function.Surjective ⇑(f.app { unop := X })
      X : C
      y : (CategoryTheory.forget D).obj (G.obj { unop := X })
      x : (CategoryTheory.forget D).obj (F.obj { unop := X })
      hx : Eq ((f.app { unop := X }) x) y
      ⊢ (CategoryTheory.Presheaf.imageSieve f y).arrows (CategoryTheory.CategoryStru …
    -/
    simpa [Presheaf.imageSieve] using ⟨x, hx⟩
    /-
      🎉 no goals
    -/


lemma extensiveTopology.isLocallySurjective_iff [FinitaryExtensive C]
    {F G : Sheaf (extensiveTopology C) D} (f : F ⟶ G)
      [PreservesFiniteProducts (forget D)] : IsLocallySurjective f ↔
        ∀ (X : C), Function.Surjective (f.val.app (op X)) :=
  extensiveTopology.presheafIsLocallySurjective_iff _ f.val


lemma regularTopology.isLocallySurjective_sheaf_of_types [Preregular C] [FinitaryPreExtensive C]
    {F G : Cᵒᵖ ⥤ Type w} (f : F ⟶ G) [PreservesFiniteProducts F] [PreservesFiniteProducts G]
      (h : Presheaf.IsLocallySurjective (coherentTopology C) f) :
        Presheaf.IsLocallySurjective (regularTopology C) f where
  imageSieve_mem y := by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      h : CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.coherentTopolo …
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      ⊢ Membership.mem ((CategoryTheory.regularTopology C) U✝) (CategoryTheory.Presh …
    -/
    replace h := h.1 y
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      h : Membership.mem ((CategoryTheory.coherentTopology C) U✝) (CategoryTheory.Pr …
      ⊢ Membership.mem ((CategoryTheory.regularTopology C) U✝) (CategoryTheory.Presh …
    -/
    rw [coherentTopology.mem_sieves_iff_hasEffectiveEpiFamily] at h
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      h : Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Categ …
      ⊢ Membership.mem ((CategoryTheory.regularTopology C) U✝) (CategoryTheory.Presh …
    -/
    obtain ⟨α, _, Z, π, h, h'⟩ := h
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      ⊢ Membership.mem ((CategoryTheory.regularTopology C) U✝) (CategoryTheory.Presh …
    -/
    rw [mem_sieves_iff_hasEffectiveEpi]
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) ((Catego …
    -/
    let x : (a : α) → (F.obj ⟨Z a⟩) := fun a ↦ (h' a).choose
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) ((Catego …
    -/
    let _ : Fintype α := Fintype.ofFinite _
    let i' : ((a : α) → (F.obj ⟨Z a⟩)) ≅ (F.obj ⟨∐ Z⟩) := (Types.productIso _).symm ≪≫
      (PreservesProduct.iso F _).symm ≪≫ F.mapIso (opCoproductIsoProduct _).symm
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
      x✝ : Fintype α := Fintype.ofFinite α
      i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) ((Catego …
    -/
    refine ⟨∐ Z, Sigma.desc π, inferInstance, i'.hom x, ?_⟩
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
      x✝ : Fintype α := Fintype.ofFinite α
      i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
      ⊢ Eq ((f.app { unop := CategoryTheory.Limits.sigmaObj Z }) (i'.hom x)) ((G.map …
    -/
    have := preservesLimitsOfShape_of_equiv (Discrete.opposite α).symm G
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
      x✝ : Fintype α := Fintype.ofFinite α
      i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
      this : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory. …
      ⊢ Eq ((f.app { unop := CategoryTheory.Limits.sigmaObj Z }) (i'.hom x)) ((G.map …
    -/
    apply Concrete.isLimit_ext _ (isLimitOfPreserves G (coproductIsCoproduct Z).op)
    /-
      case intro.intro.intro.intro.intro.a
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
      x✝ : Fintype α := Fintype.ofFinite α
      i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
      this : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory. …
      ⊢ ∀ (j : Opposite (CategoryTheory.Discrete α)), Eq (((G.mapCone (CategoryTheor …
    -/
    intro ⟨⟨a⟩⟩
    simp only [Functor.comp_obj, Functor.op_obj, Discrete.functor_obj, Functor.mapCone_pt,
      Cocone.op_pt, Cofan.mk_pt, Functor.const_obj_obj, Functor.mapCone_π_app, Cocone.op_π,
      NatTrans.op_app, Cofan.mk_ι_app, Functor.mapIso_symm, Iso.symm_hom, Iso.trans_hom,
      Functor.mapIso_inv, types_comp_apply, i', ← NatTrans.naturality_apply]
    /-
      case intro.intro.intro.intro.intro.a
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
      x✝ : Fintype α := Fintype.ofFinite α
      i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
      this : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory. …
      a : α
      ⊢ Eq ((f.app { unop := Z a }) ((F.map (CategoryTheory.Limits.Sigma.ι Z a).op)  …
    -/
    have : f.app ⟨Z a⟩ (x a) = G.map (π a).op y := (h' a).choose_spec
    /-
      case intro.intro.intro.intro.intro.a
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preregular C
      inst✝² : CategoryTheory.FinitaryPreExtensive C
      F G : CategoryTheory.Functor (Opposite C) (Type w)
      f : Quiver.Hom F G
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      U✝ : C
      y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
      α : Type
      w✝ : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) U✝
      h : CategoryTheory.EffectiveEpiFamily Z π
      h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
      x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
      x✝ : Fintype α := Fintype.ofFinite α
      i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
      this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory …
      a : α
      this : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
      ⊢ Eq ((f.app { unop := Z a }) ((F.map (CategoryTheory.Limits.Sigma.ι Z a).op)  …
    -/
    convert this
      /-
        case h.e'_2.h.h.e'_1
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preregular C
        inst✝² : CategoryTheory.FinitaryPreExtensive C
        F G : CategoryTheory.Functor (Opposite C) (Type w)
        f : Quiver.Hom F G
        inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        U✝ : C
        y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
        α : Type
        w✝ : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) U✝
        h : CategoryTheory.EffectiveEpiFamily Z π
        h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
        x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
        x✝ : Fintype α := Fintype.ofFinite α
        i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
        this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory …
        a : α
        this : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
        e_1✝ : Eq ((CategoryTheory.forget (Type w)).obj (G.obj { unop := Z a })) (G.ob …
        ⊢ Eq ((F.map (CategoryTheory.Limits.Sigma.ι Z a).op) (F.map (CategoryTheory.Li …
      -/
    · change F.map _ (F.map _ _) = _
      /-
        case h.e'_2.h.h.e'_1
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preregular C
        inst✝² : CategoryTheory.FinitaryPreExtensive C
        F G : CategoryTheory.Functor (Opposite C) (Type w)
        f : Quiver.Hom F G
        inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        U✝ : C
        y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
        α : Type
        w✝ : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) U✝
        h : CategoryTheory.EffectiveEpiFamily Z π
        h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
        x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
        x✝ : Fintype α := Fintype.ofFinite α
        i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
        this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory …
        a : α
        this : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
        e_1✝ : Eq ((CategoryTheory.forget (Type w)).obj (G.obj { unop := Z a })) (G.ob …
        ⊢ Eq (F.map (CategoryTheory.Limits.Sigma.ι Z a).op (F.map (CategoryTheory.Limi …
      -/
      rw [← FunctorToTypes.map_comp_apply, opCoproductIsoProduct_inv_comp_ι, ← piComparison_comp_π]
      /-
        case h.e'_2.h.h.e'_1
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preregular C
        inst✝² : CategoryTheory.FinitaryPreExtensive C
        F G : CategoryTheory.Functor (Opposite C) (Type w)
        f : Quiver.Hom F G
        inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        U✝ : C
        y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
        α : Type
        w✝ : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) U✝
        h : CategoryTheory.EffectiveEpiFamily Z π
        h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
        x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
        x✝ : Fintype α := Fintype.ofFinite α
        i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
        this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory …
        a : α
        this : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
        e_1✝ : Eq ((CategoryTheory.forget (Type w)).obj (G.obj { unop := Z a })) (G.ob …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.piComparison F …
      -/
      change ((PreservesProduct.iso F _).hom ≫ _) _ = _
      /-
        case h.e'_2.h.h.e'_1
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preregular C
        inst✝² : CategoryTheory.FinitaryPreExtensive C
        F G : CategoryTheory.Functor (Opposite C) (Type w)
        f : Quiver.Hom F G
        inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        U✝ : C
        y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
        α : Type
        w✝ : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) U✝
        h : CategoryTheory.EffectiveEpiFamily Z π
        h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
        x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
        x✝ : Fintype α := Fintype.ofFinite α
        i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
        this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory …
        a : α
        this : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
        e_1✝ : Eq ((CategoryTheory.forget (Type w)).obj (G.obj { unop := Z a })) (G.ob …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesProdu …
      -/
      have := Types.productIso_hom_comp_eval (fun a ↦ F.obj (op (Z a))) a
      /-
        case h.e'_2.h.h.e'_1
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preregular C
        inst✝² : CategoryTheory.FinitaryPreExtensive C
        F G : CategoryTheory.Functor (Opposite C) (Type w)
        f : Quiver.Hom F G
        inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        U✝ : C
        y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
        α : Type
        w✝ : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) U✝
        h : CategoryTheory.EffectiveEpiFamily Z π
        h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
        x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
        x✝ : Fintype α := Fintype.ofFinite α
        i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
        this✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheor …
        a : α
        this✝ : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
        e_1✝ : Eq ((CategoryTheory.forget (Type w)).obj (G.obj { unop := Z a })) (G.ob …
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.pro …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesProdu …
      -/
      rw [← Iso.eq_inv_comp] at this
      /-
        case h.e'_2.h.h.e'_1
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preregular C
        inst✝² : CategoryTheory.FinitaryPreExtensive C
        F G : CategoryTheory.Functor (Opposite C) (Type w)
        f : Quiver.Hom F G
        inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        U✝ : C
        y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
        α : Type
        w✝ : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) U✝
        h : CategoryTheory.EffectiveEpiFamily Z π
        h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
        x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
        x✝ : Fintype α := Fintype.ofFinite α
        i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
        this✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheor …
        a : α
        this✝ : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
        e_1✝ : Eq ((CategoryTheory.forget (Type w)).obj (G.obj { unop := Z a })) (G.ob …
        this : Eq (fun f => f a) (CategoryTheory.CategoryStruct.comp (CategoryTheory.L …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesProdu …
      -/
      simp only [types_comp_apply, inv_hom_id_apply, congrFun this x]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preregular C
        inst✝² : CategoryTheory.FinitaryPreExtensive C
        F G : CategoryTheory.Functor (Opposite C) (Type w)
        f : Quiver.Hom F G
        inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        U✝ : C
        y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
        α : Type
        w✝ : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) U✝
        h : CategoryTheory.EffectiveEpiFamily Z π
        h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
        x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
        x✝ : Fintype α := Fintype.ofFinite α
        i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
        this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory …
        a : α
        this : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
        e_1✝ : Eq ((CategoryTheory.forget (Type w)).obj (G.obj { unop := Z a })) (G.ob …
        ⊢ Eq ((G.map (CategoryTheory.Limits.Sigma.ι Z a).op) ((G.map (CategoryTheory.L …
      -/
    · change G.map _ (G.map _ _) = _
      /-
        case h.e'_3.h
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preregular C
        inst✝² : CategoryTheory.FinitaryPreExtensive C
        F G : CategoryTheory.Functor (Opposite C) (Type w)
        f : Quiver.Hom F G
        inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        U✝ : C
        y : (CategoryTheory.forget (Type w)).obj (G.obj { unop := U✝ })
        α : Type
        w✝ : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) U✝
        h : CategoryTheory.EffectiveEpiFamily Z π
        h' : ∀ (a : α), (CategoryTheory.Presheaf.imageSieve f y).arrows (π a)
        x : (a : α) → F.obj { unop := Z a } := fun a => Exists.choose ⋯
        x✝ : Fintype α := Fintype.ofFinite α
        i' : CategoryTheory.Iso ((a : α) → F.obj { unop := Z a }) (F.obj { unop := Cat …
        this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (Opposite (CategoryTheory …
        a : α
        this : Eq (f.app { unop := Z a } (x a)) (G.map (π a).op y)
        e_1✝ : Eq ((CategoryTheory.forget (Type w)).obj (G.obj { unop := Z a })) (G.ob …
        ⊢ Eq (G.map (CategoryTheory.Limits.Sigma.ι Z a).op (G.map (CategoryTheory.Limi …
      -/
      simp only [← FunctorToTypes.map_comp_apply, ← op_comp, Sigma.ι_desc]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-11-26")] alias regularTopology.isLocallySurjective_sheafOfTypes :=
regularTopology.isLocallySurjective_sheaf_of_types


lemma coherentTopology.presheafIsLocallySurjective_iff {F G : Cᵒᵖ ⥤ D} (f : F ⟶ G)
    [Preregular C] [FinitaryPreExtensive C] [PreservesFiniteProducts F] [PreservesFiniteProducts G]
      [PreservesFiniteProducts (forget D)] :
        Presheaf.IsLocallySurjective (coherentTopology C) f ↔
          Presheaf.IsLocallySurjective (regularTopology C) f := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.ConcreteCategory D
    F G : CategoryTheory.Functor (Opposite C) D
    f : Quiver.Hom F G
    inst✝⁴ : CategoryTheory.Preregular C
    inst✝³ : CategoryTheory.FinitaryPreExtensive C
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
    ⊢ Iff (CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.coherentTop …
  -/
  constructor
  · rw [Presheaf.isLocallySurjective_iff_whisker_forget,
      Presheaf.isLocallySurjective_iff_whisker_forget (J := regularTopology C)]
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝⁴ : CategoryTheory.Preregular C
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.coherentTopology …
    -/
    exact regularTopology.isLocallySurjective_sheaf_of_types _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝⁴ : CategoryTheory.Preregular C
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      ⊢ CategoryTheory.Presheaf.IsLocallySurjective (CategoryTheory.regularTopology  …
    -/
  · refine Presheaf.isLocallySurjective_of_le (J := regularTopology C) ?_ _
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝⁴ : CategoryTheory.Preregular C
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      ⊢ LE.le (CategoryTheory.regularTopology C) (CategoryTheory.coherentTopology C)
    -/
    rw [← extensive_regular_generate_coherent]
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁵ : CategoryTheory.ConcreteCategory D
      F G : CategoryTheory.Functor (Opposite C) D
      f : Quiver.Hom F G
      inst✝⁴ : CategoryTheory.Preregular C
      inst✝³ : CategoryTheory.FinitaryPreExtensive C
      inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
      inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts G
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget D)
      ⊢ LE.le (CategoryTheory.regularTopology C) (CategoryTheory.Coverage.toGrothend …
    -/
    exact (Coverage.gi _).gc.monotone_l le_sup_right
    /-
      🎉 no goals
    -/


lemma coherentTopology.isLocallySurjective_iff [Preregular C] [FinitaryExtensive C]
    {F G : Sheaf (coherentTopology C) D} (f : F ⟶ G) [PreservesFiniteProducts (forget D)] :
      IsLocallySurjective f ↔ Presheaf.IsLocallySurjective (regularTopology C) f.val :=
  presheafIsLocallySurjective_iff _ f.val


