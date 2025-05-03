/-- A subpresheaf of a presheaf consists of a subset of `F.obj U` for every `U`,
compatible with the restriction maps `F.map i`. -/
@[ext]
structure Subpresheaf (F : Cᵒᵖ ⥤ Type w) where
  /-- If `G` is a sub-presheaf of `F`, then the sections of `G` on `U` forms a subset of sections of
    `F` on `U`. -/
  obj : ∀ U, Set (F.obj U)
  /-- If `G` is a sub-presheaf of `F` and `i : U ⟶ V`, then for each `G`-sections on `U` `x`,
    `F i x` is in `F(V)`. -/
  map : ∀ {U V : Cᵒᵖ} (i : U ⟶ V), obj U ⊆ F.map i ⁻¹' obj V


instance : PartialOrder (Subpresheaf F) :=
  PartialOrder.lift Subpresheaf.obj (fun _ _ => Subpresheaf.ext)


instance : Top (Subpresheaf F) :=
                                     /-
                                       C : Type u
                                       inst✝ : CategoryTheory.Category.{v, u} C
                                       J : CategoryTheory.GrothendieckTopology C
                                       F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
                                       G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
                                       U x✝² : Opposite C
                                       x✝¹ : Quiver.Hom U x✝²
                                       x : F.obj U
                                       x✝ : Membership.mem ((fun x => Top.top) U) x
                                       ⊢ Membership.mem (Set.preimage (F.map x✝¹) ((fun x => Top.top) x✝²)) x
                                     -/
  ⟨⟨fun _ => ⊤, @fun U _ _ x _ => by aesop_cat⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


instance : Nonempty (Subpresheaf F) :=
  inferInstance


/-- The subpresheaf as a presheaf. -/
@[simps!]
def Subpresheaf.toPresheaf : Cᵒᵖ ⥤ Type w where
  obj U := G.obj U
  map := @fun _ _ i x => ⟨F.map i x, G.map i x.prop⟩
  map_id X := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      X : Opposite C
      ⊢ Eq ({ obj := fun U => ↑(G.obj U), map := fun x x_1 i x_2 => ⟨F.map i ↑x_2, ⋯ …
    -/
    ext ⟨x, _⟩
    /-
      case h.mk.a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      X : Opposite C
      x : F.obj X
      property✝ : Membership.mem (G.obj X) x
      ⊢ Eq ↑({ obj := fun U => ↑(G.obj U), map := fun x x_1 i x_2 => ⟨F.map i ↑x_2,  …
    -/
    dsimp
    /-
      case h.mk.a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      X : Opposite C
      x : F.obj X
      property✝ : Membership.mem (G.obj X) x
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id X) x) x
    -/
    simp only [FunctorToTypes.map_id_apply]
    /-
      🎉 no goals
    -/
  map_comp := @fun X Y Z i j => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      X Y Z : Opposite C
      i : Quiver.Hom X Y
      j : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun U => ↑(G.obj U), map := fun x x_1 i x_2 => ⟨F.map i ↑x_2, ⋯ …
    -/
    ext ⟨x, _⟩
    /-
      case h.mk.a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      X Y Z : Opposite C
      i : Quiver.Hom X Y
      j : Quiver.Hom Y Z
      x : F.obj X
      property✝ : Membership.mem (G.obj X) x
      ⊢ Eq ↑({ obj := fun U => ↑(G.obj U), map := fun x x_1 i x_2 => ⟨F.map i ↑x_2,  …
    -/
    dsimp
    /-
      case h.mk.a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      X Y Z : Opposite C
      i : Quiver.Hom X Y
      j : Quiver.Hom Y Z
      x : F.obj X
      property✝ : Membership.mem (G.obj X) x
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp i j) x) (F.map j (F.map i x))
    -/
    simp only [FunctorToTypes.map_comp_apply]
    /-
      🎉 no goals
    -/


instance {U} : CoeHead (G.toPresheaf.obj U) (F.obj U) where
  coe := Subtype.val


/-- The inclusion of a subpresheaf to the original presheaf. -/
@[simps]
def Subpresheaf.ι : G.toPresheaf ⟶ F where app _ x := x


instance : Mono G.ι :=
  ⟨@fun _ _ _ e =>
    NatTrans.ext <|
      funext fun U => funext fun x => Subtype.ext <| congr_fun (congr_app e U) x⟩


/-- The inclusion of a subpresheaf to a larger subpresheaf -/
@[simps]
def Subpresheaf.homOfLe {G G' : Subpresheaf F} (h : G ≤ G') : G.toPresheaf ⟶ G'.toPresheaf where
  app U x := ⟨x, h U x.prop⟩


instance {G G' : Subpresheaf F} (h : G ≤ G') : Mono (Subpresheaf.homOfLe h) :=
  ⟨fun _ _ e =>
    NatTrans.ext <|
      funext fun U =>
        funext fun x =>
          Subtype.ext <| (congr_arg Subtype.val <| (congr_fun (congr_app e U) x : _) : _)⟩


@[reassoc (attr := simp)]
theorem Subpresheaf.homOfLe_ι {G G' : Subpresheaf F} (h : G ≤ G') :
    Subpresheaf.homOfLe h ≫ G'.ι = G.ι := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
  -/
  ext
  /-
    case w.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    x✝ : Opposite C
    a✝ : G.toPresheaf.obj x✝
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : IsIso (Subpresheaf.ι (⊤ : Subpresheaf F)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    ⊢ CategoryTheory.IsIso Top.top.ι
  -/
  refine @NatIso.isIso_of_isIso_app _ _ _ _ _ _ _ ?_
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    ⊢ ∀ (X : Opposite C), CategoryTheory.IsIso (Top.top.ι.app X)
  -/
  intro X
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    X : Opposite C
    ⊢ CategoryTheory.IsIso (Top.top.ι.app X)
  -/
  rw [isIso_iff_bijective]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    X : Opposite C
    ⊢ Function.Bijective (Top.top.ι.app X)
  -/
  exact ⟨Subtype.coe_injective, fun x => ⟨⟨x, _root_.trivial⟩, rfl⟩⟩
  /-
    🎉 no goals
  -/


theorem Subpresheaf.eq_top_iff_isIso : G = ⊤ ↔ IsIso G.ι := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    ⊢ Iff (Eq G Top.top) (CategoryTheory.IsIso G.ι)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      G : CategoryTheory.GrothendieckTopology.Subpresheaf F
      ⊢ Eq G Top.top → CategoryTheory.IsIso G.ι
    -/
  · rintro rfl
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      ⊢ CategoryTheory.IsIso Top.top.ι
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      G : CategoryTheory.GrothendieckTopology.Subpresheaf F
      ⊢ CategoryTheory.IsIso G.ι → Eq G Top.top
    -/
  · intro H
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      G : CategoryTheory.GrothendieckTopology.Subpresheaf F
      H : CategoryTheory.IsIso G.ι
      ⊢ Eq G Top.top
    -/
    ext U x
    /-
      case mpr.obj.h.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      G : CategoryTheory.GrothendieckTopology.Subpresheaf F
      H : CategoryTheory.IsIso G.ι
      U : Opposite C
      x : F.obj U
      ⊢ Iff (Membership.mem (G.obj U) x) (Membership.mem (Top.top.obj U) x)
    -/
    apply (iff_of_eq (iff_true _)).mpr
    /-
      case mpr.obj.h.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      G : CategoryTheory.GrothendieckTopology.Subpresheaf F
      H : CategoryTheory.IsIso G.ι
      U : Opposite C
      x : F.obj U
      ⊢ Membership.mem (G.obj U) x
    -/
    rw [← IsIso.inv_hom_id_apply (G.ι.app U) x]
    /-
      case mpr.obj.h.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      G : CategoryTheory.GrothendieckTopology.Subpresheaf F
      H : CategoryTheory.IsIso G.ι
      U : Opposite C
      x : F.obj U
      ⊢ Membership.mem (G.obj U) ((G.ι.app U) ((CategoryTheory.inv (G.ι.app U)) x))
    -/
    exact ((inv (G.ι.app U)) x).2
    /-
      🎉 no goals
    -/


/-- If the image of a morphism falls in a subpresheaf, then the morphism factors through it. -/
@[simps!]
def Subpresheaf.lift (f : F' ⟶ F) (hf : ∀ U x, f.app U x ∈ G.obj U) : F' ⟶ G.toPresheaf where
  app U x := ⟨f.app U x, hf U x⟩
  naturality := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom F' F
      hf : ∀ (U : Opposite C) (x : F'.obj U), Membership.mem (G.obj U) (f.app U x)
      ⊢ ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStru …
    -/
    have := elementwise_of% f.naturality
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom F' F
      hf : ∀ (U : Opposite C) (x : F'.obj U), Membership.mem (G.obj U) (f.app U x)
      this : ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y) (x : F'.obj X), Eq (f.app Y …
      ⊢ ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStru …
    -/
    intros
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom F' F
      hf : ∀ (U : Opposite C) (x : F'.obj U), Membership.mem (G.obj U) (f.app U x)
      this : ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y) (x : F'.obj X), Eq (f.app Y …
      X✝ Y✝ : Opposite C
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.map f✝) ((fun U x => ⟨f.app U x,  …
    -/
    refine funext fun x => Subtype.ext ?_
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom F' F
      hf : ∀ (U : Opposite C) (x : F'.obj U), Membership.mem (G.obj U) (f.app U x)
      this : ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y) (x : F'.obj X), Eq (f.app Y …
      X✝ Y✝ : Opposite C
      f✝ : Quiver.Hom X✝ Y✝
      x : F'.obj X✝
      ⊢ Eq ↑(CategoryTheory.CategoryStruct.comp (F'.map f✝) ((fun U x => ⟨f.app U x, …
    -/
    simp only [toPresheaf_obj, types_comp_apply]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom F' F
      hf : ∀ (U : Opposite C) (x : F'.obj U), Membership.mem (G.obj U) (f.app U x)
      this : ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y) (x : F'.obj X), Eq (f.app Y …
      X✝ Y✝ : Opposite C
      f✝ : Quiver.Hom X✝ Y✝
      x : F'.obj X✝
      ⊢ Eq (f.app Y✝ (F'.map f✝ x)) ↑(G.toPresheaf.map f✝ ⟨f.app X✝ x, ⋯⟩)
    -/
    exact this _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem Subpresheaf.lift_ι (f : F' ⟶ F) (hf : ∀ U x, f.app U x ∈ G.obj U) :
    G.lift f hf ≫ G.ι = f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    f : Quiver.Hom F' F
    hf : ∀ (U : Opposite C) (x : F'.obj U), Membership.mem (G.obj U) (f.app U x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.lift f hf) G.ι) f
  -/
  ext
  /-
    case w.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    f : Quiver.Hom F' F
    hf : ∀ (U : Opposite C) (x : F'.obj U), Membership.mem (G.obj U) (f.app U x)
    x✝ : Opposite C
    a✝ : F'.obj x✝
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (G.lift f hf) G.ι).app x✝ a✝) (f.app …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a subpresheaf `G` of `F`, an `F`-section `s` on `U`, we may define a sieve of `U`
consisting of all `f : V ⟶ U` such that the restriction of `s` along `f` is in `G`. -/
@[simps]
def Subpresheaf.sieveOfSection {U : Cᵒᵖ} (s : F.obj U) : Sieve (unop U) where
  arrows V f := F.map f.op s ∈ G.obj (op V)
  downward_closed := @fun V W i hi j => by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      U : Opposite C
      s : F.obj U
      V W : C
      i : Quiver.Hom V (Opposite.unop U)
      hi : (fun V f => Membership.mem (G.obj { unop := V }) (F.map f.op s)) V i
      j : Quiver.Hom W V
      ⊢ (fun V f => Membership.mem (G.obj { unop := V }) (F.map f.op s)) W (Category …
    -/
    simp only [op_unop, op_comp, FunctorToTypes.map_comp_apply]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      U : Opposite C
      s : F.obj U
      V W : C
      i : Quiver.Hom V (Opposite.unop U)
      hi : (fun V f => Membership.mem (G.obj { unop := V }) (F.map f.op s)) V i
      j : Quiver.Hom W V
      ⊢ Membership.mem (G.obj { unop := W }) (F.map j.op (F.map i.op s))
    -/
    exact G.map _ hi
    /-
      🎉 no goals
    -/


/-- Given an `F`-section `s` on `U` and a subpresheaf `G`, we may define a family of elements in
`G` consisting of the restrictions of `s` -/
def Subpresheaf.familyOfElementsOfSection {U : Cᵒᵖ} (s : F.obj U) :
    (G.sieveOfSection s).1.FamilyOfElements G.toPresheaf := fun _ i hi => ⟨F.map i.op s, hi⟩


theorem Subpresheaf.family_of_elements_compatible {U : Cᵒᵖ} (s : F.obj U) :
    (G.familyOfElementsOfSection s).Compatible := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    ⊢ (G.familyOfElementsOfSection s).Compatible
  -/
  intro Y₁ Y₂ Z g₁ g₂ f₁ f₂ h₁ h₂ e
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ (Opposite.unop U)
    f₂ : Quiver.Hom Y₂ (Opposite.unop U)
    h₁ : (G.sieveOfSection s).arrows f₁
    h₂ : (G.sieveOfSection s).arrows f₂
    e : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStru …
    ⊢ Eq (G.toPresheaf.map g₁.op (G.familyOfElementsOfSection s f₁ h₁)) (G.toPresh …
  -/
  refine Subtype.ext ?_ -- Porting note: `ext1` does not work here
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ (Opposite.unop U)
    f₂ : Quiver.Hom Y₂ (Opposite.unop U)
    h₁ : (G.sieveOfSection s).arrows f₁
    h₂ : (G.sieveOfSection s).arrows f₂
    e : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStru …
    ⊢ Eq ↑(G.toPresheaf.map g₁.op (G.familyOfElementsOfSection s f₁ h₁)) ↑(G.toPre …
  -/
  change F.map g₁.op (F.map f₁.op s) = F.map g₂.op (F.map f₂.op s)
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ (Opposite.unop U)
    f₂ : Quiver.Hom Y₂ (Opposite.unop U)
    h₁ : (G.sieveOfSection s).arrows f₁
    h₂ : (G.sieveOfSection s).arrows f₂
    e : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStru …
    ⊢ Eq (F.map g₁.op (F.map f₁.op s)) (F.map g₂.op (F.map f₂.op s))
  -/
  rw [← FunctorToTypes.map_comp_apply, ← FunctorToTypes.map_comp_apply, ← op_comp, ← op_comp, e]
  /-
    🎉 no goals
  -/


theorem Subpresheaf.nat_trans_naturality (f : F' ⟶ G.toPresheaf) {U V : Cᵒᵖ} (i : U ⟶ V)
    (x : F'.obj U) : (f.app V (F'.map i x)).1 = F.map i (f.app U x).1 :=
  congr_arg Subtype.val (FunctorToTypes.naturality _ _ f i x)


/-- The sheafification of a subpresheaf as a subpresheaf.
Note that this is a sheaf only when the whole presheaf is a sheaf. -/
def Subpresheaf.sheafify : Subpresheaf F where
  obj U := { s | G.sieveOfSection s ∈ J (unop U) }
  map := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      ⊢ ∀ {U V : Opposite C} (i : Quiver.Hom U V), HasSubset.Subset ((fun U => setOf …
    -/
    rintro U V i s hs
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      U V : Opposite C
      i : Quiver.Hom U V
      s : F.obj U
      hs : Membership.mem ((fun U => setOf fun s => Membership.mem (J (Opposite.unop …
      ⊢ Membership.mem (Set.preimage (F.map i) ((fun U => setOf fun s => Membership. …
    -/
    refine J.superset_covering ?_ (J.pullback_stable i.unop hs)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      U V : Opposite C
      i : Quiver.Hom U V
      s : F.obj U
      hs : Membership.mem ((fun U => setOf fun s => Membership.mem (J (Opposite.unop …
      ⊢ LE.le (CategoryTheory.Sieve.pullback i.unop (G.sieveOfSection s)) (G.sieveOf …
    -/
    intro _ _ h
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      U V : Opposite C
      i : Quiver.Hom U V
      s : F.obj U
      hs : Membership.mem ((fun U => setOf fun s => Membership.mem (J (Opposite.unop …
      Y✝ : C
      f✝ : Quiver.Hom Y✝ (Opposite.unop V)
      h : (CategoryTheory.Sieve.pullback i.unop (G.sieveOfSection s)).arrows f✝
      ⊢ (G.sieveOfSection (F.map i s)).arrows f✝
    -/
    dsimp at h ⊢
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      U V : Opposite C
      i : Quiver.Hom U V
      s : F.obj U
      hs : Membership.mem ((fun U => setOf fun s => Membership.mem (J (Opposite.unop …
      Y✝ : C
      f✝ : Quiver.Hom Y✝ (Opposite.unop V)
      h : Membership.mem (G.obj { unop := Y✝ }) (F.map (CategoryTheory.CategoryStruc …
      ⊢ Membership.mem (G.obj { unop := Y✝ }) (F.map f✝.op (F.map i s))
    -/
    rwa [← FunctorToTypes.map_comp_apply]
    /-
      🎉 no goals
    -/


theorem Subpresheaf.le_sheafify : G ≤ G.sheafify J := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    ⊢ LE.le G (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G)
  -/
  intro U s hs
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    hs : Membership.mem (G.obj U) s
    ⊢ Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J  …
  -/
  change _ ∈ J _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    hs : Membership.mem (G.obj U) s
    ⊢ Membership.mem (J (Opposite.unop U)) (G.sieveOfSection s)
  -/
  convert J.top_mem U.unop -- Porting note: `U.unop` can not be inferred now
  /-
    case h.e'_5
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    hs : Membership.mem (G.obj U) s
    ⊢ Eq (G.sieveOfSection s) Top.top
  -/
  rw [eq_top_iff]
  /-
    case h.e'_5
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    hs : Membership.mem (G.obj U) s
    ⊢ LE.le Top.top (G.sieveOfSection s)
  -/
  rintro V i -
  /-
    case h.e'_5
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    U : Opposite C
    s : F.obj U
    hs : Membership.mem (G.obj U) s
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    ⊢ (G.sieveOfSection s).arrows i
  -/
  exact G.map i.op hs
  /-
    🎉 no goals
  -/


theorem Subpresheaf.eq_sheafify (h : Presieve.IsSheaf J F) (hG : Presieve.IsSheaf J G.toPresheaf) :
    G = G.sheafify J := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F
    hG : CategoryTheory.Presieve.IsSheaf J G.toPresheaf
    ⊢ Eq G (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G)
  -/
  apply (G.le_sheafify J).antisymm
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F
    hG : CategoryTheory.Presieve.IsSheaf J G.toPresheaf
    ⊢ LE.le (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G) G
  -/
  intro U s hs
  suffices ((hG _ hs).amalgamate _ (G.family_of_elements_compatible s)).1 = s by
    rw [← this]
    exact ((hG _ hs).amalgamate _ (G.family_of_elements_compatible s)).2
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F
    hG : CategoryTheory.Presieve.IsSheaf J G.toPresheaf
    U : Opposite C
    s : F.obj U
    hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ Eq (↑(⋯.amalgamate (G.familyOfElementsOfSection s) ⋯)) s
  -/
  apply (h _ hs).isSeparatedFor.ext
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F
    hG : CategoryTheory.Presieve.IsSheaf J G.toPresheaf
    U : Opposite C
    s : F.obj U
    hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y (Opposite.unop U)⦄, (G.sieveOfSection s).arrows  …
  -/
  intro V i hi
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F
    hG : CategoryTheory.Presieve.IsSheaf J G.toPresheaf
    U : Opposite C
    s : F.obj U
    hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    hi : (G.sieveOfSection s).arrows i
    ⊢ Eq (F.map i.op ↑(⋯.amalgamate (G.familyOfElementsOfSection s) ⋯)) (F.map i.o …
  -/
  exact (congr_arg Subtype.val ((hG _ hs).valid_glue (G.family_of_elements_compatible s) _ hi) : _)
  /-
    🎉 no goals
  -/


theorem Subpresheaf.sheafify_isSheaf (hF : Presieve.IsSheaf J F) :
    Presieve.IsSheaf J (G.sheafify J).toPresheaf := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    ⊢ CategoryTheory.Presieve.IsSheaf J (CategoryTheory.GrothendieckTopology.Subpr …
  -/
  intro U S hS x hx
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  let S' := Sieve.bind S fun Y f hf => G.sieveOfSection (x f hf).1
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  have := fun (V) (i : V ⟶ U) (hi : S' i) => hi
  -- Porting note: change to explicit variable so that `choose` can find the correct
  -- dependent functions. Thus everything follows need two additional explicit variables.
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    this : ∀ (V : C) (i : Quiver.Hom V U), S'.arrows i → S'.arrows i
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  choose W i₁ i₂ hi₂ h₁ h₂ using this
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), (fun x_1 x_2 h => ((fu …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  dsimp [-Sieve.bind_apply] at *
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Membership.mem (G.obj  …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  let x'' : Presieve.FamilyOfElements F S' := fun V i hi => F.map (i₁ V i hi).op (x _ (hi₂ V i hi))
  have H : ∀ s, x.IsAmalgamation s ↔ x''.IsAmalgamation s.1 := by
    intro s
    constructor
    · intro H V i hi
      dsimp only [x'']
      conv_lhs => rw [← h₂ _ _ hi]
      rw [← H _ (hi₂ _ _ hi)]
      exact FunctorToTypes.map_comp_apply F (i₂ _ _ hi).op (i₁ _ _ hi).op _
    · intro H V i hi
      refine Subtype.ext ?_
      apply (hF _ (x i hi).2).isSeparatedFor.ext
      intro V' i' hi'
      have hi'' : S' (i' ≫ i) := ⟨_, _, _, hi, hi', rfl⟩
      have := H _ hi''
      rw [op_comp, F.map_comp] at this
      exact this.trans (congr_arg Subtype.val (hx _ _ (hi₂ _ _ hi'') hi (h₂ _ _ hi'')))
  have : x''.Compatible := by
    intro V₁ V₂ V₃ g₁ g₂ g₃ g₄ S₁ S₂ e
    rw [← FunctorToTypes.map_comp_apply, ← FunctorToTypes.map_comp_apply]
    exact
      congr_arg Subtype.val
        (hx (g₁ ≫ i₁ _ _ S₁) (g₂ ≫ i₁ _ _ S₂) (hi₂ _ _ S₁) (hi₂ _ _ S₂)
        (by simp only [Category.assoc, h₂, e]))
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Membership.mem (G.obj  …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    x'' : CategoryTheory.Presieve.FamilyOfElements F S'.arrows := fun V i hi => F. …
    H : ∀ (s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPr …
    this : x''.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  obtain ⟨t, ht, ht'⟩ := hF _ (J.bind_covering hS fun V i hi => (x i hi).2) _ this
  /-
    case intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Membership.mem (G.obj  …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    x'' : CategoryTheory.Presieve.FamilyOfElements F S'.arrows := fun V i hi => F. …
    H : ∀ (s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPr …
    this : x''.Compatible
    t : F.obj { unop := U }
    ht : x''.IsAmalgamation t
    ht' : ∀ (y : F.obj { unop := U }), (fun t => x''.IsAmalgamation t) y → Eq y t
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  refine ⟨⟨t, _⟩, (H ⟨t, ?_⟩).mpr ht, fun y hy => Subtype.ext (ht' _ ((H _).mp hy))⟩
  /-
    case intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Membership.mem (G.obj  …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    x'' : CategoryTheory.Presieve.FamilyOfElements F S'.arrows := fun V i hi => F. …
    H : ∀ (s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPr …
    this : x''.Compatible
    t : F.obj { unop := U }
    ht : x''.IsAmalgamation t
    ht' : ∀ (y : F.obj { unop := U }), (fun t => x''.IsAmalgamation t) y → Eq y t
    ⊢ Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J  …
  -/
  refine J.superset_covering ?_ (J.bind_covering hS fun V i hi => (x i hi).2)
  /-
    case intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Membership.mem (G.obj  …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    x'' : CategoryTheory.Presieve.FamilyOfElements F S'.arrows := fun V i hi => F. …
    H : ∀ (s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPr …
    this : x''.Compatible
    t : F.obj { unop := U }
    ht : x''.IsAmalgamation t
    ht' : ∀ (y : F.obj { unop := U }), (fun t => x''.IsAmalgamation t) y → Eq y t
    ⊢ LE.le (CategoryTheory.Sieve.bind S.arrows fun V i hi => G.sieveOfSection ↑(x …
  -/
  intro V i hi
  /-
    case intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Membership.mem (G.obj  …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    x'' : CategoryTheory.Presieve.FamilyOfElements F S'.arrows := fun V i hi => F. …
    H : ∀ (s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPr …
    this : x''.Compatible
    t : F.obj { unop := U }
    ht : x''.IsAmalgamation t
    ht' : ∀ (y : F.obj { unop := U }), (fun t => x''.IsAmalgamation t) y → Eq y t
    V : C
    i : Quiver.Hom V (Opposite.unop { unop := U })
    hi : (CategoryTheory.Sieve.bind S.arrows fun V i hi => G.sieveOfSection ↑(x i  …
    ⊢ (G.sieveOfSection t).arrows i
  -/
  dsimp
  /-
    case intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Membership.mem (G.obj  …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    x'' : CategoryTheory.Presieve.FamilyOfElements F S'.arrows := fun V i hi => F. …
    H : ∀ (s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPr …
    this : x''.Compatible
    t : F.obj { unop := U }
    ht : x''.IsAmalgamation t
    ht' : ∀ (y : F.obj { unop := U }), (fun t => x''.IsAmalgamation t) y → Eq y t
    V : C
    i : Quiver.Hom V (Opposite.unop { unop := U })
    hi : (CategoryTheory.Sieve.bind S.arrows fun V i hi => G.sieveOfSection ↑(x i  …
    ⊢ Membership.mem (G.obj { unop := V }) (F.map i.op t)
  -/
  rw [ht _ hi]
  /-
    case intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    hF : CategoryTheory.Presieve.IsSheaf J F
    U : C
    S : CategoryTheory.Sieve U
    hS : Membership.mem (J U) S
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.GrothendieckTopol …
    hx : x.Compatible
    S' : CategoryTheory.Sieve U := CategoryTheory.Sieve.bind S.arrows fun Y f hf = …
    W : (V : C) → (i : Quiver.Hom V U) → S'.arrows i → C
    i₁ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom V (W V i …
    i₂ : (V : C) → (i : Quiver.Hom V U) → (hi : S'.arrows i) → Quiver.Hom (W V i h …
    hi₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), S.arrows (i₂ V i hi)
    h₁ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Membership.mem (G.obj  …
    h₂ : ∀ (V : C) (i : Quiver.Hom V U) (hi : S'.arrows i), Eq (CategoryTheory.Cat …
    x'' : CategoryTheory.Presieve.FamilyOfElements F S'.arrows := fun V i hi => F. …
    H : ∀ (s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPr …
    this : x''.Compatible
    t : F.obj { unop := U }
    ht : x''.IsAmalgamation t
    ht' : ∀ (y : F.obj { unop := U }), (fun t => x''.IsAmalgamation t) y → Eq y t
    V : C
    i : Quiver.Hom V (Opposite.unop { unop := U })
    hi : (CategoryTheory.Sieve.bind S.arrows fun V i hi => G.sieveOfSection ↑(x i  …
    ⊢ Membership.mem (G.obj { unop := V }) (x'' i hi)
  -/
  exact h₁ _ _ hi
  /-
    🎉 no goals
  -/


theorem Subpresheaf.eq_sheafify_iff (h : Presieve.IsSheaf J F) :
    G = G.sheafify J ↔ Presieve.IsSheaf J G.toPresheaf :=
  ⟨fun e => e.symm ▸ G.sheafify_isSheaf h, G.eq_sheafify h⟩


theorem Subpresheaf.isSheaf_iff (h : Presieve.IsSheaf J F) :
    Presieve.IsSheaf J G.toPresheaf ↔
      ∀ (U) (s : F.obj U), G.sieveOfSection s ∈ J (unop U) → s ∈ G.obj U := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F
    ⊢ Iff (CategoryTheory.Presieve.IsSheaf J G.toPresheaf) (∀ (U : Opposite C) (s  …
  -/
  rw [← G.eq_sheafify_iff h]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F
    ⊢ Iff (Eq G (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G)) (∀ …
  -/
  change _ ↔ G.sheafify J ≤ G
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F
    ⊢ Iff (Eq G (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G)) (L …
  -/
  exact ⟨Eq.ge, (G.le_sheafify J).antisymm⟩
  /-
    🎉 no goals
  -/


theorem Subpresheaf.sheafify_sheafify (h : Presieve.IsSheaf J F) :
    (G.sheafify J).sheafify J = G.sheafify J :=
  ((Subpresheaf.eq_sheafify_iff _ h).mpr <| G.sheafify_isSheaf h).symm


/-- The lift of a presheaf morphism onto the sheafification subpresheaf. -/
noncomputable def Subpresheaf.sheafifyLift (f : G.toPresheaf ⟶ F') (h : Presieve.IsSheaf J F') :
    (G.sheafify J).toPresheaf ⟶ F' where
  app _ s := (h (G.sieveOfSection s.1) s.prop).amalgamate
    (_) ((G.family_of_elements_compatible s.1).compPresheafMap f)
  naturality := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom G.toPresheaf F'
      h : CategoryTheory.Presieve.IsSheaf J F'
      ⊢ ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStru …
    -/
    intro U V i
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom G.toPresheaf F'
      h : CategoryTheory.Presieve.IsSheaf J F'
      U V : Opposite C
      i : Quiver.Hom U V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GrothendieckTopology …
    -/
    ext s
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom G.toPresheaf F'
      h : CategoryTheory.Presieve.IsSheaf J F'
      U V : Opposite C
      i : Quiver.Hom U V
      s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GrothendieckTopology …
    -/
    apply (h _ ((Subpresheaf.sheafify J G).toPresheaf.map i s).prop).isSeparatedFor.ext
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom G.toPresheaf F'
      h : CategoryTheory.Presieve.IsSheaf J F'
      U V : Opposite C
      i : Quiver.Hom U V
      s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
      ⊢ ∀ ⦃Y : C⦄ ⦃f_1 : Quiver.Hom Y (Opposite.unop V)⦄, (G.sieveOfSection ↑((Categ …
    -/
    intro W j hj
    refine (Presieve.IsSheafFor.valid_glue (h _ ((G.sheafify J).toPresheaf.map i s).2)
      ((G.family_of_elements_compatible _).compPresheafMap _) _ hj).trans ?_
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom G.toPresheaf F'
      h : CategoryTheory.Presieve.IsSheaf J F'
      U V : Opposite C
      i : Quiver.Hom U V
      s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
      W : C
      j : Quiver.Hom W (Opposite.unop V)
      hj : (G.sieveOfSection ↑((CategoryTheory.GrothendieckTopology.Subpresheaf.shea …
      ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.compPresheafMap f (G.familyOfEl …
    -/
    dsimp
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom G.toPresheaf F'
      h : CategoryTheory.Presieve.IsSheaf J F'
      U V : Opposite C
      i : Quiver.Hom U V
      s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
      W : C
      j : Quiver.Hom W (Opposite.unop V)
      hj : (G.sieveOfSection ↑((CategoryTheory.GrothendieckTopology.Subpresheaf.shea …
      ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.compPresheafMap f (G.familyOfEl …
    -/
    conv_rhs => rw [← FunctorToTypes.map_comp_apply]
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom G.toPresheaf F'
      h : CategoryTheory.Presieve.IsSheaf J F'
      U V : Opposite C
      i : Quiver.Hom U V
      s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
      W : C
      j : Quiver.Hom W (Opposite.unop V)
      hj : (G.sieveOfSection ↑((CategoryTheory.GrothendieckTopology.Subpresheaf.shea …
      ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.compPresheafMap f (G.familyOfEl …
    -/
    change _ = F'.map (j ≫ i.unop).op _
    refine Eq.trans ?_ (Presieve.IsSheafFor.valid_glue (h _ s.2)
      ((G.family_of_elements_compatible s.1).compPresheafMap f) (j ≫ i.unop) ?_).symm
      /-
        case h.refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
        G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
        f : Quiver.Hom G.toPresheaf F'
        h : CategoryTheory.Presieve.IsSheaf J F'
        U V : Opposite C
        i : Quiver.Hom U V
        s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
        W : C
        j : Quiver.Hom W (Opposite.unop V)
        hj : (G.sieveOfSection ↑((CategoryTheory.GrothendieckTopology.Subpresheaf.shea …
        ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.compPresheafMap f (G.familyOfEl …
      -/
    · dsimp [Presieve.FamilyOfElements.compPresheafMap]
      /-
        case h.refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
        G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
        f : Quiver.Hom G.toPresheaf F'
        h : CategoryTheory.Presieve.IsSheaf J F'
        U V : Opposite C
        i : Quiver.Hom U V
        s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
        W : C
        j : Quiver.Hom W (Opposite.unop V)
        hj : (G.sieveOfSection ↑((CategoryTheory.GrothendieckTopology.Subpresheaf.shea …
        ⊢ Eq (f.app { unop := W } (G.familyOfElementsOfSection (F.map i ↑s) j hj)) (f. …
      -/
      exact congr_arg _ (Subtype.ext (FunctorToTypes.map_comp_apply _ _ _ _).symm)
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
        G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
        f : Quiver.Hom G.toPresheaf F'
        h : CategoryTheory.Presieve.IsSheaf J F'
        U V : Opposite C
        i : Quiver.Hom U V
        s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
        W : C
        j : Quiver.Hom W (Opposite.unop V)
        hj : (G.sieveOfSection ↑((CategoryTheory.GrothendieckTopology.Subpresheaf.shea …
        ⊢ (G.sieveOfSection ↑s).arrows (CategoryTheory.CategoryStruct.comp j i.unop)
      -/
    · dsimp [Presieve.FamilyOfElements.compPresheafMap] at hj ⊢
      /-
        case h.refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
        G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
        f : Quiver.Hom G.toPresheaf F'
        h : CategoryTheory.Presieve.IsSheaf J F'
        U V : Opposite C
        i : Quiver.Hom U V
        s : (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G).toPresheaf. …
        W : C
        j : Quiver.Hom W (Opposite.unop V)
        hj : Membership.mem (G.obj { unop := W }) (F.map j.op (F.map i ↑s))
        ⊢ Membership.mem (G.obj { unop := W }) (F.map (CategoryTheory.CategoryStruct.c …
      -/
      rwa [FunctorToTypes.map_comp_apply]
      /-
        🎉 no goals
      -/


theorem Subpresheaf.to_sheafifyLift (f : G.toPresheaf ⟶ F') (h : Presieve.IsSheaf J F') :
    Subpresheaf.homOfLe (G.le_sheafify J) ≫ G.sheafifyLift f h = f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    f : Quiver.Hom G.toPresheaf F'
    h : CategoryTheory.Presieve.IsSheaf J F'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
  -/
  ext U s
  /-
    case w.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    f : Quiver.Hom G.toPresheaf F'
    h : CategoryTheory.Presieve.IsSheaf J F'
    U : Opposite C
    s : G.toPresheaf.obj U
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology …
  -/
  apply (h _ ((Subpresheaf.homOfLe (G.le_sheafify J)).app U s).prop).isSeparatedFor.ext
  /-
    case w.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    f : Quiver.Hom G.toPresheaf F'
    h : CategoryTheory.Presieve.IsSheaf J F'
    U : Opposite C
    s : G.toPresheaf.obj U
    ⊢ ∀ ⦃Y : C⦄ ⦃f_1 : Quiver.Hom Y (Opposite.unop U)⦄, (G.sieveOfSection ↑((Categ …
  -/
  intro V i hi
  /-
    case w.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    f : Quiver.Hom G.toPresheaf F'
    h : CategoryTheory.Presieve.IsSheaf J F'
    U : Opposite C
    s : G.toPresheaf.obj U
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    hi : (G.sieveOfSection ↑((CategoryTheory.GrothendieckTopology.Subpresheaf.homO …
    ⊢ Eq (F'.map i.op ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Grothen …
  -/
  have := elementwise_of% f.naturality
  -- Porting note: filled in some underscores where Lean3 could automatically fill.
  exact (Presieve.IsSheafFor.valid_glue (h _ ((homOfLe (_ : G ≤ sheafify J G)).app U s).2)
    ((G.family_of_elements_compatible _).compPresheafMap _) _ hi).trans (this _ _)


theorem Subpresheaf.to_sheafify_lift_unique (h : Presieve.IsSheaf J F')
    (l₁ l₂ : (G.sheafify J).toPresheaf ⟶ F')
    (e : Subpresheaf.homOfLe (G.le_sheafify J) ≫ l₁ = Subpresheaf.homOfLe (G.le_sheafify J) ≫ l₂) :
    l₁ = l₂ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F'
    l₁ l₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J …
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    ⊢ Eq l₁ l₂
  -/
  ext U ⟨s, hs⟩
  /-
    case w.h.h.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F'
    l₁ l₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J …
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F.obj U
    hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ Eq (l₁.app U ⟨s, hs⟩) (l₂.app U ⟨s, hs⟩)
  -/
  apply (h _ hs).isSeparatedFor.ext
  /-
    case w.h.h.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F'
    l₁ l₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J …
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F.obj U
    hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y (Opposite.unop U)⦄, (G.sieveOfSection s).arrows  …
  -/
  rintro V i hi
  /-
    case w.h.h.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F'
    l₁ l₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J …
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F.obj U
    hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    hi : (G.sieveOfSection s).arrows i
    ⊢ Eq (F'.map i.op (l₁.app U ⟨s, hs⟩)) (F'.map i.op (l₂.app U ⟨s, hs⟩))
  -/
  dsimp at hi
  /-
    case w.h.h.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F'
    l₁ l₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J …
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F.obj U
    hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    hi : Membership.mem (G.obj { unop := V }) (F.map i.op s)
    ⊢ Eq (F'.map i.op (l₁.app U ⟨s, hs⟩)) (F'.map i.op (l₂.app U ⟨s, hs⟩))
  -/
  rw [← FunctorToTypes.naturality, ← FunctorToTypes.naturality]
  /-
    case w.h.h.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Functor (Opposite C) (Type w)
    G : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : CategoryTheory.Presieve.IsSheaf J F'
    l₁ l₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J …
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F.obj U
    hs : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    hi : Membership.mem (G.obj { unop := V }) (F.map i.op s)
    ⊢ Eq (l₁.app { unop := V } ((CategoryTheory.GrothendieckTopology.Subpresheaf.s …
  -/
  exact (congr_fun (congr_app e <| op V) ⟨_, hi⟩ : _)
  /-
    🎉 no goals
  -/


theorem Subpresheaf.sheafify_le (h : G ≤ G') (hF : Presieve.IsSheaf J F)
    (hG' : Presieve.IsSheaf J G'.toPresheaf) : G.sheafify J ≤ G' := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    hF : CategoryTheory.Presieve.IsSheaf J F
    hG' : CategoryTheory.Presieve.IsSheaf J G'.toPresheaf
    ⊢ LE.le (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J G) G'
  -/
  intro U x hx
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    hF : CategoryTheory.Presieve.IsSheaf J F
    hG' : CategoryTheory.Presieve.IsSheaf J G'.toPresheaf
    U : Opposite C
    x : F.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ Membership.mem (G'.obj U) x
  -/
  convert ((G.sheafifyLift (Subpresheaf.homOfLe h) hG').app U ⟨x, hx⟩).2
  /-
    case h.e'_5
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    hF : CategoryTheory.Presieve.IsSheaf J F
    hG' : CategoryTheory.Presieve.IsSheaf J G'.toPresheaf
    U : Opposite C
    x : F.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ Eq x ↑((G.sheafifyLift (CategoryTheory.GrothendieckTopology.Subpresheaf.homO …
  -/
  apply (hF _ hx).isSeparatedFor.ext
  /-
    case h.e'_5
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    hF : CategoryTheory.Presieve.IsSheaf J F
    hG' : CategoryTheory.Presieve.IsSheaf J G'.toPresheaf
    U : Opposite C
    x : F.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y (Opposite.unop U)⦄, (G.sieveOfSection x).arrows  …
  -/
  intro V i hi
  have :=
    congr_arg (fun f : G.toPresheaf ⟶ G'.toPresheaf => (NatTrans.app f (op V) ⟨_, hi⟩).1)
      (G.to_sheafifyLift (Subpresheaf.homOfLe h) hG')
  /-
    case h.e'_5
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    hF : CategoryTheory.Presieve.IsSheaf J F
    hG' : CategoryTheory.Presieve.IsSheaf J G'.toPresheaf
    U : Opposite C
    x : F.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    hi : (G.sieveOfSection x).arrows i
    this : Eq ((fun f => ↑(f.app { unop := V } ⟨F.map i.op x, hi⟩)) (CategoryTheor …
    ⊢ Eq (F.map i.op x) (F.map i.op ↑((G.sheafifyLift (CategoryTheory.Grothendieck …
  -/
  convert this.symm
  /-
    case h.e'_3
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    hF : CategoryTheory.Presieve.IsSheaf J F
    hG' : CategoryTheory.Presieve.IsSheaf J G'.toPresheaf
    U : Opposite C
    x : F.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    hi : (G.sieveOfSection x).arrows i
    this : Eq ((fun f => ↑(f.app { unop := V } ⟨F.map i.op x, hi⟩)) (CategoryTheor …
    ⊢ Eq (F.map i.op ↑((G.sheafifyLift (CategoryTheory.GrothendieckTopology.Subpre …
  -/
  rw [← Subpresheaf.nat_trans_naturality]
  /-
    case h.e'_3
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
    h : LE.le G G'
    hF : CategoryTheory.Presieve.IsSheaf J F
    hG' : CategoryTheory.Presieve.IsSheaf J G'.toPresheaf
    U : Opposite C
    x : F.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    hi : (G.sieveOfSection x).arrows i
    this : Eq ((fun f => ↑(f.app { unop := V } ⟨F.map i.op x, hi⟩)) (CategoryTheor …
    ⊢ Eq (↑((G.sheafifyLift (CategoryTheory.GrothendieckTopology.Subpresheaf.homOf …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The image presheaf of a morphism, whose components are the set-theoretic images. -/
@[simps]
def imagePresheaf (f : F' ⟶ F) : Subpresheaf F where
  obj U := Set.range (f.app U)
  map := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom F' F
      ⊢ ∀ {U V : Opposite C} (i : Quiver.Hom U V), HasSubset.Subset ((fun U => Set.r …
    -/
    rintro U V i _ ⟨x, rfl⟩
    /-
      case intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom F' F
      U V : Opposite C
      i : Quiver.Hom U V
      x : F'.obj U
      ⊢ Membership.mem (Set.preimage (F.map i) ((fun U => Set.range (f.app U)) V)) ( …
    -/
    have := elementwise_of% f.naturality
    /-
      case intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F F' F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F
      f : Quiver.Hom F' F
      U V : Opposite C
      i : Quiver.Hom U V
      x : F'.obj U
      this : ∀ ⦃X Y : Opposite C⦄ (f_1 : Quiver.Hom X Y) (x : F'.obj X), Eq (f.app Y …
      ⊢ Membership.mem (Set.preimage (F.map i) ((fun U => Set.range (f.app U)) V)) ( …
    -/
    exact ⟨_, this i x⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem top_subpresheaf_obj (U) : (⊤ : Subpresheaf F).obj U = ⊤ :=
  rfl


@[simp]
theorem imagePresheaf_id : imagePresheaf (𝟙 F) = ⊤ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    ⊢ Eq (CategoryTheory.GrothendieckTopology.imagePresheaf (CategoryTheory.Catego …
  -/
  ext
  /-
    case obj.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    x✝¹ : Opposite C
    x✝ : F.obj x✝¹
    ⊢ Iff (Membership.mem ((CategoryTheory.GrothendieckTopology.imagePresheaf (Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A morphism factors through the image presheaf. -/
@[simps!]
def toImagePresheaf (f : F' ⟶ F) : F' ⟶ (imagePresheaf f).toPresheaf :=
  (imagePresheaf f).lift f fun _ _ => Set.mem_range_self _


/-- A morphism factors through the sheafification of the image presheaf. -/
@[simps!]
def toImagePresheafSheafify (f : F' ⟶ F) : F' ⟶ ((imagePresheaf f).sheafify J).toPresheaf :=
  toImagePresheaf f ≫ Subpresheaf.homOfLe ((imagePresheaf f).le_sheafify J)


@[reassoc (attr := simp)]
theorem toImagePresheaf_ι (f : F' ⟶ F) : toImagePresheaf f ≫ (imagePresheaf f).ι = f :=
  (imagePresheaf f).lift_ι _ _


theorem imagePresheaf_comp_le (f₁ : F ⟶ F') (f₂ : F' ⟶ F'') :
    imagePresheaf (f₁ ≫ f₂) ≤ imagePresheaf f₂ := fun U _ hx => ⟨f₁.app U hx.choose, hx.choose_spec⟩


instance isIso_toImagePresheaf {F F' : Cᵒᵖ ⥤ (Type (max v w))} (f : F ⟶ F') [hf : Mono f] :
  IsIso (toImagePresheaf f) := by
  have : ∀ (X : Cᵒᵖ), IsIso ((toImagePresheaf f).app X) := by
    intro X
    rw [isIso_iff_bijective]
    constructor
    · intro x y e
      have := (NatTrans.mono_iff_mono_app f).mp hf X
      rw [mono_iff_injective] at this
      exact this (congr_arg Subtype.val e : _)
    · rintro ⟨_, ⟨x, rfl⟩⟩
      exact ⟨x, rfl⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Functor (Opposite C) (Type (max v w))
    f : Quiver.Hom F F'
    hf : CategoryTheory.Mono f
    this : ∀ (X : Opposite C), CategoryTheory.IsIso ((CategoryTheory.GrothendieckT …
    ⊢ CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePresheaf f)
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


/-- The image sheaf of a morphism between sheaves, defined to be the sheafification of
`image_presheaf`. -/
@[simps]
def imageSheaf {F F' : Sheaf J (Type w)} (f : F ⟶ F') : Sheaf J (Type w) :=
  ⟨((imagePresheaf f.1).sheafify J).toPresheaf, by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
      F F' : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F F'
      ⊢ CategoryTheory.Presheaf.IsSheaf J (CategoryTheory.GrothendieckTopology.Subpr …
    -/
    rw [isSheaf_iff_isSheaf_of_type]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
      F F' : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F F'
      ⊢ CategoryTheory.Presieve.IsSheaf J (CategoryTheory.GrothendieckTopology.Subpr …
    -/
    apply Subpresheaf.sheafify_isSheaf
    /-
      case hF
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
      F F' : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F F'
      ⊢ CategoryTheory.Presieve.IsSheaf J F'.val
    -/
    rw [← isSheaf_iff_isSheaf_of_type]
    /-
      case hF
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
      G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
      F F' : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F F'
      ⊢ CategoryTheory.Presheaf.IsSheaf J F'.val
    -/
    exact F'.2⟩
    /-
      🎉 no goals
    -/


/-- A morphism factors through the image sheaf. -/
@[simps]
def toImageSheaf {F F' : Sheaf J (Type w)} (f : F ⟶ F') : F ⟶ imageSheaf f :=
  ⟨toImagePresheafSheafify J f.1⟩


/-- The inclusion of the image sheaf to the target. -/
@[simps]
def imageSheafι {F F' : Sheaf J (Type w)} (f : F ⟶ F') : imageSheaf f ⟶ F' :=
  ⟨Subpresheaf.ι _⟩


@[reassoc (attr := simp)]
theorem toImageSheaf_ι {F F' : Sheaf J (Type w)} (f : F ⟶ F') :
    toImageSheaf f ≫ imageSheafι f = f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
  -/
  ext1
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
  -/
  simp [toImagePresheafSheafify]
  /-
    🎉 no goals
  -/


instance {F F' : Sheaf J (Type w)} (f : F ⟶ F') : Mono (imageSheafι f) :=
  (sheafToPresheaf J _).mono_of_mono_map
    (by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
        G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
        F F' : CategoryTheory.Sheaf J (Type w)
        f : Quiver.Hom F F'
        ⊢ CategoryTheory.Mono ((CategoryTheory.sheafToPresheaf J (Type w)).map (Catego …
      -/
      dsimp
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
        G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
        F F' : CategoryTheory.Sheaf J (Type w)
        f : Quiver.Hom F F'
        ⊢ CategoryTheory.Mono (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafif …
      -/
      infer_instance)
      /-
        🎉 no goals
      -/


instance {F F' : Sheaf J (Type w)} (f : F ⟶ F') : Epi (toImageSheaf f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    ⊢ CategoryTheory.Epi (CategoryTheory.GrothendieckTopology.toImageSheaf f)
  -/
  refine ⟨@fun G' g₁ g₂ e => ?_⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G'✝ : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    G' : CategoryTheory.Sheaf J (Type w)
    g₁ g₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.imageSheaf f) G'
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    ⊢ Eq g₁ g₂
  -/
  ext U ⟨s, hx⟩
  /-
    case h.w.h.h.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G'✝ : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    G' : CategoryTheory.Sheaf J (Type w)
    g₁ g₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.imageSheaf f) G'
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F'.val.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ Eq (g₁.val.app U ⟨s, hx⟩) (g₂.val.app U ⟨s, hx⟩)
  -/
  apply ((isSheaf_iff_isSheaf_of_type J _).mp G'.2 _ hx).isSeparatedFor.ext
  /-
    case h.w.h.h.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G'✝ : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    G' : CategoryTheory.Sheaf J (Type w)
    g₁ g₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.imageSheaf f) G'
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F'.val.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    ⊢ ∀ ⦃Y : C⦄ ⦃f_1 : Quiver.Hom Y (Opposite.unop U)⦄, ((CategoryTheory.Grothendi …
  -/
  rintro V i ⟨y, e'⟩
  /-
    case h.w.h.h.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G'✝ : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    G' : CategoryTheory.Sheaf J (Type w)
    g₁ g₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.imageSheaf f) G'
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F'.val.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    y : F.val.obj { unop := V }
    e' : Eq (f.val.app { unop := V } y) (F'.val.map i.op s)
    ⊢ Eq (G'.val.map i.op (g₁.val.app U ⟨s, hx⟩)) (G'.val.map i.op (g₂.val.app U ⟨ …
  -/
  change (g₁.val.app _ ≫ G'.val.map _) _ = (g₂.val.app _ ≫ G'.val.map _) _
  /-
    case h.w.h.h.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G'✝ : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    G' : CategoryTheory.Sheaf J (Type w)
    g₁ g₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.imageSheaf f) G'
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F'.val.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    y : F.val.obj { unop := V }
    e' : Eq (f.val.app { unop := V } y) (F'.val.map i.op s)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (g₁.val.app { unop := Opposite.unop U …
  -/
  rw [← NatTrans.naturality, ← NatTrans.naturality]
  have E : (toImageSheaf f).val.app (op V) y = (imageSheaf f).val.map i.op ⟨s, hx⟩ :=
    Subtype.ext e'
  /-
    case h.w.h.h.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G'✝ : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    G' : CategoryTheory.Sheaf J (Type w)
    g₁ g₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.imageSheaf f) G'
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F'.val.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    y : F.val.obj { unop := V }
    e' : Eq (f.val.app { unop := V } y) (F'.val.map i.op s)
    E : Eq ((CategoryTheory.GrothendieckTopology.toImageSheaf f).val.app { unop := …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GrothendieckTopology …
  -/
  have := congr_arg (fun f : F ⟶ G' => (Sheaf.Hom.val f).app _ y) e
  /-
    case h.w.h.h.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G'✝ : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    G' : CategoryTheory.Sheaf J (Type w)
    g₁ g₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.imageSheaf f) G'
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F'.val.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    y : F.val.obj { unop := V }
    e' : Eq (f.val.app { unop := V } y) (F'.val.map i.op s)
    E : Eq ((CategoryTheory.GrothendieckTopology.toImageSheaf f).val.app { unop := …
    this : Eq ((fun f => f.val.app { unop := V } y) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GrothendieckTopology …
  -/
  dsimp at this ⊢
  /-
    case h.w.h.h.mk.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
    G G'✝ : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
    F F' : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F F'
    G' : CategoryTheory.Sheaf J (Type w)
    g₁ g₂ : Quiver.Hom (CategoryTheory.GrothendieckTopology.imageSheaf f) G'
    e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopolog …
    U : Opposite C
    s : F'.val.obj U
    hx : Membership.mem ((CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify …
    V : C
    i : Quiver.Hom V (Opposite.unop U)
    y : F.val.obj { unop := V }
    e' : Eq (f.val.app { unop := V } y) (F'.val.map i.op s)
    E : Eq ((CategoryTheory.GrothendieckTopology.toImageSheaf f).val.app { unop := …
    this : Eq (g₁.val.app { unop := V } ((J.toImagePresheafSheafify f.val).app { u …
    ⊢ Eq (g₁.val.app { unop := V } ((CategoryTheory.GrothendieckTopology.Subpreshe …
  -/
                   /-
                     🎉 no goals
                   -/
  convert this <;> exact E.symm
                   /-
                     🎉 no goals
                   -/


/-- The mono factorization given by `image_sheaf` for a morphism. -/
def imageMonoFactorization {F F' : Sheaf J (Type w)} (f : F ⟶ F') : Limits.MonoFactorisation f where
  I := imageSheaf f
  m := imageSheafι f
  e := toImageSheaf f


/-- The mono factorization given by `image_sheaf` for a morphism is an image. -/
noncomputable def imageFactorization {F F' : Sheaf J (Type (max v u))} (f : F ⟶ F') :
    Limits.ImageFactorisation f where
  F := imageMonoFactorization f
  isImage :=
    { lift := fun I => by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Quiver.Hom (CategoryTheory.GrothendieckTopology.imageMonoFactorization f).I  …
        -/
        haveI M := (Sheaf.Hom.mono_iff_presheaf_mono J (Type (max v u)) _).mp I.m_mono
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          M : CategoryTheory.Mono I.m.val
          ⊢ Quiver.Hom (CategoryTheory.GrothendieckTopology.imageMonoFactorization f).I  …
        -/
        haveI := isIso_toImagePresheaf I.m.1
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          M : CategoryTheory.Mono I.m.val
          this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
          ⊢ Quiver.Hom (CategoryTheory.GrothendieckTopology.imageMonoFactorization f).I  …
        -/
        refine ⟨Subpresheaf.homOfLe ?_ ≫ inv (toImagePresheaf I.m.1)⟩
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          M : CategoryTheory.Mono I.m.val
          this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
          ⊢ LE.le (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J (CategoryT …
        -/
        apply Subpresheaf.sheafify_le
          /-
            case h
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
            G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
            F F' : CategoryTheory.Sheaf J (Type (max v u))
            f : Quiver.Hom F F'
            I : CategoryTheory.Limits.MonoFactorisation f
            M : CategoryTheory.Mono I.m.val
            this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
            ⊢ LE.le (CategoryTheory.GrothendieckTopology.imagePresheaf f.val) (CategoryThe …
          -/
        · conv_lhs => rw [← I.fac]
          /-
            case h
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
            G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
            F F' : CategoryTheory.Sheaf J (Type (max v u))
            f : Quiver.Hom F F'
            I : CategoryTheory.Limits.MonoFactorisation f
            M : CategoryTheory.Mono I.m.val
            this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
            ⊢ LE.le (CategoryTheory.GrothendieckTopology.imagePresheaf (CategoryTheory.Cat …
          -/
          apply imagePresheaf_comp_le
          /-
            🎉 no goals
          -/
          /-
            case hF
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
            G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
            F F' : CategoryTheory.Sheaf J (Type (max v u))
            f : Quiver.Hom F F'
            I : CategoryTheory.Limits.MonoFactorisation f
            M : CategoryTheory.Mono I.m.val
            this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
            ⊢ CategoryTheory.Presieve.IsSheaf J F'.val
          -/
        · rw [← isSheaf_iff_isSheaf_of_type]
          /-
            case hF
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
            G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
            F F' : CategoryTheory.Sheaf J (Type (max v u))
            f : Quiver.Hom F F'
            I : CategoryTheory.Limits.MonoFactorisation f
            M : CategoryTheory.Mono I.m.val
            this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
            ⊢ CategoryTheory.Presheaf.IsSheaf J F'.val
          -/
          exact F'.2
          /-
            🎉 no goals
          -/
          /-
            case hG'
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
            G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
            F F' : CategoryTheory.Sheaf J (Type (max v u))
            f : Quiver.Hom F F'
            I : CategoryTheory.Limits.MonoFactorisation f
            M : CategoryTheory.Mono I.m.val
            this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
            ⊢ CategoryTheory.Presieve.IsSheaf J (CategoryTheory.GrothendieckTopology.image …
          -/
        · apply Presieve.isSheaf_iso J (asIso <| toImagePresheaf I.m.1)
          /-
            case hG'
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
            G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
            F F' : CategoryTheory.Sheaf J (Type (max v u))
            f : Quiver.Hom F F'
            I : CategoryTheory.Limits.MonoFactorisation f
            M : CategoryTheory.Mono I.m.val
            this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
            ⊢ CategoryTheory.Presieve.IsSheaf J I.I.val
          -/
          rw [← isSheaf_iff_isSheaf_of_type]
          /-
            case hG'
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
            G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
            F F' : CategoryTheory.Sheaf J (Type (max v u))
            f : Quiver.Hom F F'
            I : CategoryTheory.Limits.MonoFactorisation f
            M : CategoryTheory.Mono I.m.val
            this : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshe …
            ⊢ CategoryTheory.Presheaf.IsSheaf J I.I.val
          -/
          exact I.I.2
          /-
            🎉 no goals
          -/
      lift_fac := fun I => by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun I => { val := CategoryTheory.Ca …
        -/
        ext1
        /-
          case h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun I => { val := CategoryTheory.Ca …
        -/
        dsimp [imageMonoFactorization]
        /-
          case h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        generalize_proofs h
        /-
          case h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          h : LE.le (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J (Categor …
          pf✝ : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshea …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [← Subpresheaf.homOfLe_ι h, Category.assoc]
        /-
          case h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          h : LE.le (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J (Categor …
          pf✝ : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshea …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GrothendieckTopology. …
        -/
        congr 1
        /-
          case h.e_a
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          F✝ F'✝ F'' : CategoryTheory.Functor (Opposite C) (Type w)
          G G' : CategoryTheory.GrothendieckTopology.Subpresheaf F✝
          F F' : CategoryTheory.Sheaf J (Type (max v u))
          f : Quiver.Hom F F'
          I : CategoryTheory.Limits.MonoFactorisation f
          h : LE.le (CategoryTheory.GrothendieckTopology.Subpresheaf.sheafify J (Categor …
          pf✝ : CategoryTheory.IsIso (CategoryTheory.GrothendieckTopology.toImagePreshea …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.G …
        -/
        rw [IsIso.inv_comp_eq, toImagePresheaf_ι] }
        /-
          🎉 no goals
        -/


instance : Limits.HasImages (Sheaf J (Type max v u)) :=
  ⟨@fun _ _ f => ⟨⟨imageFactorization f⟩⟩⟩


