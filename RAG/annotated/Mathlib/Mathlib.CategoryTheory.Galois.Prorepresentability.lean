/-- A pointed Galois object is a Galois object with a fixed point of its fiber. -/
structure PointedGaloisObject (F : C ⥤ FintypeCat.{w}) : Type (max u₁ u₂ w) where
  /-- The underlying object of `C`. -/
  obj : C
  /-- An element of the fiber of `obj`. -/
  pt : F.obj obj
  /-- `obj` is Galois. -/
  isGalois : IsGalois obj := by infer_instance


instance (X : PointedGaloisObject F) : CoeDep (PointedGaloisObject F) X C where
  coe := X.obj


variable {F} in
/-- The type of homomorphisms between two pointed Galois objects. This is a homomorphism
of the underlying objects of `C` that maps the distinguished points to each other. -/
@[ext]
structure Hom (A B : PointedGaloisObject F) where
  /-- The underlying homomorphism of `C`. -/
  val : A.obj ⟶ B.obj
  /-- The distinguished point of `A` is mapped to the distinguished point of `B`. -/
  comp : F.map val A.pt = B.pt := by simp


/-- The category of pointed Galois objects. -/
instance : Category.{u₂} (PointedGaloisObject F) where
  Hom A B := Hom A B
  id A := { val := 𝟙 (A : C) }
  comp {A B C} f g := { val := f.val ≫ g.val }


instance {A B : PointedGaloisObject F} : Coe (Hom A B) (A.obj ⟶ B.obj) where
  coe f := f.val


@[ext]
lemma hom_ext {A B : PointedGaloisObject F} {f g : A ⟶ B} (h : f.val = g.val) : f = g :=
  Hom.ext h


@[simp]
lemma id_val (A : PointedGaloisObject F) : 𝟙 A = 𝟙 A.obj :=
  rfl


@[simp, reassoc]
lemma comp_val {A B C : PointedGaloisObject F} (f : A ⟶ B) (g : B ⟶ C) :
    (f ≫ g).val = f.val ≫ g.val :=
  rfl


/-- The canonical functor from pointed Galois objects to `C`. -/
def incl : PointedGaloisObject F ⥤ C where
  obj := fun A ↦ A
  map := fun ⟨f, _⟩ ↦ f


@[simp]
lemma incl_obj (A : PointedGaloisObject F) : (incl F).obj A = A :=
  rfl


@[simp]
lemma incl_map {A B : PointedGaloisObject F} (f : A ⟶ B) : (incl F).map f = f.val :=
  rfl


/-- `F ⋙ FintypeCat.incl` as a cocone over `(can F).op ⋙ coyoneda`.
This is a colimit cocone (see `PreGaloisCategory.isColimìt`) -/
def cocone : Cocone ((incl F).op ⋙ coyoneda) where
  pt := F ⋙ FintypeCat.incl
  ι := {
    app := fun ⟨A, a, _⟩ ↦ { app := fun X (f : (A : C) ⟶ X) ↦ F.map f a }
    naturality := fun ⟨A, a, _⟩ ⟨B, b, _⟩ ⟨f, (hf : F.map f b = a)⟩ ↦ by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
        inst✝ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        x✝² x✝¹ : Opposite (CategoryTheory.PreGaloisCategory.PointedGaloisObject F)
        A : C
        a : ↑(F.obj A)
        isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
        B : C
        b : ↑(F.obj B)
        isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
        x✝ : Quiver.Hom { unop := { obj := A, pt := a, isGalois := isGalois✝¹ } } { un …
        f : Quiver.Hom (Opposite.unop { unop := { obj := B, pt := b, isGalois := isGal …
        hf : Eq (F.map f b) a
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.PreGaloisCategory.P …
      -/
      ext Y (g : (A : C) ⟶ Y)
      /-
        case w.h.h
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
        inst✝ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        x✝² x✝¹ : Opposite (CategoryTheory.PreGaloisCategory.PointedGaloisObject F)
        A : C
        a : ↑(F.obj A)
        isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
        B : C
        b : ↑(F.obj B)
        isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
        x✝ : Quiver.Hom { unop := { obj := A, pt := a, isGalois := isGalois✝¹ } } { un …
        f : Quiver.Hom (Opposite.unop { unop := { obj := B, pt := b, isGalois := isGal …
        hf : Eq (F.map f b) a
        Y : C
        g : Quiver.Hom A Y
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.PreGaloisCategory. …
      -/
      suffices h : F.map g (F.map f b) = F.map g a by simpa
      /-
        case w.h.h
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
        inst✝ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        x✝² x✝¹ : Opposite (CategoryTheory.PreGaloisCategory.PointedGaloisObject F)
        A : C
        a : ↑(F.obj A)
        isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
        B : C
        b : ↑(F.obj B)
        isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
        x✝ : Quiver.Hom { unop := { obj := A, pt := a, isGalois := isGalois✝¹ } } { un …
        f : Quiver.Hom (Opposite.unop { unop := { obj := B, pt := b, isGalois := isGal …
        hf : Eq (F.map f b) a
        Y : C
        g : Quiver.Hom A Y
        ⊢ Eq (F.map g (F.map f b)) (F.map g a)
      -/
      rw [hf]
      /-
        🎉 no goals
      -/
  }


@[simp]
lemma cocone_app (A : PointedGaloisObject F) (B : C) (f : (A : C) ⟶ B) :
    ((cocone F).ι.app ⟨A⟩).app B f = F.map f A.pt :=
  rfl


/-- The category of pointed Galois objects is cofiltered. -/
instance : IsCofilteredOrEmpty (PointedGaloisObject F) where
  cone_objs := fun ⟨A, a, _⟩ ⟨B, b, _⟩ ↦ by
    obtain ⟨Z, f, z, hgal, hfz⟩ := exists_hom_from_galois_of_fiber F (A ⨯ B)
      <| (fiberBinaryProductEquiv F A B).symm (a, b)
    /-
      case intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      x✝¹ x✝ : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      A : C
      a : ↑(F.obj A)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      Z : C
      f : Quiver.Hom Z (CategoryTheory.Limits.prod A B)
      z : ↑(F.obj Z)
      hgal : CategoryTheory.PreGaloisCategory.IsGalois Z
      hfz : Eq (F.map f z) ((CategoryTheory.PreGaloisCategory.fiberBinaryProductEqui …
      ⊢ Exists fun W => Exists fun x => Exists fun x => True
    -/
    refine ⟨⟨Z, z, hgal⟩, ⟨f ≫ prod.fst, ?_⟩, ⟨f ≫ prod.snd, ?_⟩, trivial⟩
      /-
        case intro.intro.intro.intro.refine_1
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        inst✝¹ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        x✝¹ x✝ : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
        A : C
        a : ↑(F.obj A)
        isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
        B : C
        b : ↑(F.obj B)
        isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
        Z : C
        f : Quiver.Hom Z (CategoryTheory.Limits.prod A B)
        z : ↑(F.obj Z)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois Z
        hfz : Eq (F.map f z) ((CategoryTheory.PreGaloisCategory.fiberBinaryProductEqui …
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f CategoryTheory.Limits.prod.f …
      -/
    · simp only [F.map_comp, hfz, FintypeCat.comp_apply, fiberBinaryProductEquiv_symm_fst_apply]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        inst✝¹ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        x✝¹ x✝ : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
        A : C
        a : ↑(F.obj A)
        isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
        B : C
        b : ↑(F.obj B)
        isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
        Z : C
        f : Quiver.Hom Z (CategoryTheory.Limits.prod A B)
        z : ↑(F.obj Z)
        hgal : CategoryTheory.PreGaloisCategory.IsGalois Z
        hfz : Eq (F.map f z) ((CategoryTheory.PreGaloisCategory.fiberBinaryProductEqui …
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f CategoryTheory.Limits.prod.s …
      -/
    · simp only [F.map_comp, hfz, FintypeCat.comp_apply, fiberBinaryProductEquiv_symm_snd_apply]
      /-
        🎉 no goals
      -/
  cone_maps := fun ⟨A, a, _⟩ ⟨B, b, _⟩ ⟨f, hf⟩ ⟨g, hg⟩ ↦ by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      x✝³ x✝² : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      A : C
      a : ↑(F.obj A)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      x✝¹ x✝ : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ } { obj := B,  …
      f : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ }.obj { obj := B, p …
      hf : Eq (F.map f { obj := A, pt := a, isGalois := isGalois✝¹ }.pt) { obj := B, …
      g : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ }.obj { obj := B, p …
      hg : Eq (F.map g { obj := A, pt := a, isGalois := isGalois✝¹ }.pt) { obj := B, …
      ⊢ Exists fun W => Exists fun h => Eq (CategoryTheory.CategoryStruct.comp h { v …
    -/
    obtain ⟨Z, h, z, hgal, hhz⟩ := exists_hom_from_galois_of_fiber F A a
    /-
      case intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      x✝³ x✝² : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      A : C
      a : ↑(F.obj A)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      x✝¹ x✝ : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ } { obj := B,  …
      f : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ }.obj { obj := B, p …
      hf : Eq (F.map f { obj := A, pt := a, isGalois := isGalois✝¹ }.pt) { obj := B, …
      g : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ }.obj { obj := B, p …
      hg : Eq (F.map g { obj := A, pt := a, isGalois := isGalois✝¹ }.pt) { obj := B, …
      Z : C
      h : Quiver.Hom Z A
      z : ↑(F.obj Z)
      hgal : CategoryTheory.PreGaloisCategory.IsGalois Z
      hhz : Eq (F.map h z) a
      ⊢ Exists fun W => Exists fun h => Eq (CategoryTheory.CategoryStruct.comp h { v …
    -/
    refine ⟨⟨Z, z, hgal⟩, ⟨h, hhz⟩, hom_ext ?_⟩
    /-
      case intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      x✝³ x✝² : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      A : C
      a : ↑(F.obj A)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      x✝¹ x✝ : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ } { obj := B,  …
      f : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ }.obj { obj := B, p …
      hf : Eq (F.map f { obj := A, pt := a, isGalois := isGalois✝¹ }.pt) { obj := B, …
      g : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ }.obj { obj := B, p …
      hg : Eq (F.map g { obj := A, pt := a, isGalois := isGalois✝¹ }.pt) { obj := B, …
      Z : C
      h : Quiver.Hom Z A
      z : ↑(F.obj Z)
      hgal : CategoryTheory.PreGaloisCategory.IsGalois Z
      hhz : Eq (F.map h z) a
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := h, comp := hhz } { val := f, …
    -/
    apply evaluation_injective_of_isConnected F Z B z
    /-
      case intro.intro.intro.intro.a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      x✝³ x✝² : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      A : C
      a : ↑(F.obj A)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois B
      x✝¹ x✝ : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ } { obj := B,  …
      f : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ }.obj { obj := B, p …
      hf : Eq (F.map f { obj := A, pt := a, isGalois := isGalois✝¹ }.pt) { obj := B, …
      g : Quiver.Hom { obj := A, pt := a, isGalois := isGalois✝¹ }.obj { obj := B, p …
      hg : Eq (F.map g { obj := A, pt := a, isGalois := isGalois✝¹ }.pt) { obj := B, …
      Z : C
      h : Quiver.Hom Z A
      z : ↑(F.obj Z)
      hgal : CategoryTheory.PreGaloisCategory.IsGalois Z
      hhz : Eq (F.map h z) a
      ⊢ Eq ((fun f => F.map f z) (CategoryTheory.CategoryStruct.comp { val := h, com …
    -/
    simp [hhz, hf, hg]
    /-
      🎉 no goals
    -/


/-- `cocone F` is a colimit cocone, i.e. `F` is pro-represented by `incl F`. -/
noncomputable def isColimit : IsColimit (cocone F) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.PreGaloisCategory.PointedGal …
  -/
  refine evaluationJointlyReflectsColimits _ (fun X ↦ ?_)
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation C (Type u₂)).ob …
  -/
  refine Types.FilteredColimit.isColimitOf _ _ ?_ ?_
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      ⊢ ∀ (x : (((CategoryTheory.evaluation C (Type u₂)).obj X).mapCocone (CategoryT …
    -/
  · intro (x : F.obj X)
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      x : ↑(F.obj X)
      ⊢ Exists fun i => Exists fun xi => Eq x ((((CategoryTheory.evaluation C (Type  …
    -/
    obtain ⟨Y, i, y, h1, _, _⟩ := fiber_in_connected_component F X x
    /-
      case refine_1.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      x : ↑(F.obj X)
      Y : C
      i : Quiver.Hom Y X
      y : ↑(F.obj Y)
      h1 : Eq (F.map i y) x
      left✝ : CategoryTheory.PreGaloisCategory.IsConnected Y
      right✝ : CategoryTheory.Mono i
      ⊢ Exists fun i => Exists fun xi => Eq x ((((CategoryTheory.evaluation C (Type  …
    -/
    obtain ⟨Z, f, z, hgal, hfz⟩ := exists_hom_from_galois_of_fiber F Y y
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      x : ↑(F.obj X)
      Y : C
      i : Quiver.Hom Y X
      y : ↑(F.obj Y)
      h1 : Eq (F.map i y) x
      left✝ : CategoryTheory.PreGaloisCategory.IsConnected Y
      right✝ : CategoryTheory.Mono i
      Z : C
      f : Quiver.Hom Z Y
      z : ↑(F.obj Z)
      hgal : CategoryTheory.PreGaloisCategory.IsGalois Z
      hfz : Eq (F.map f z) y
      ⊢ Exists fun i => Exists fun xi => Eq x ((((CategoryTheory.evaluation C (Type  …
    -/
    refine ⟨⟨Z, z, hgal⟩, f ≫ i, ?_⟩
    simp only [mapCocone_ι_app, evaluation_obj_map, cocone_app, map_comp,
      ← h1, FintypeCat.comp_apply, hfz]
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      ⊢ ∀ (i j : Opposite (CategoryTheory.PreGaloisCategory.PointedGaloisObject F))  …
    -/
  · intro ⟨A, a, _⟩ ⟨B, b, _⟩ (u : (A : C) ⟶ X) (v : (B : C) ⟶ X) (h : F.map u a = F.map v b)
    obtain ⟨⟨Z, z, _⟩, ⟨f, hf⟩, ⟨g, hg⟩, _⟩ :=
      IsFilteredOrEmpty.cocone_objs (C := (PointedGaloisObject F)ᵒᵖ)
        ⟨{ obj := A, pt := a}⟩ ⟨{obj := B, pt := b}⟩
    /-
      case refine_2.intro.op.mk.intro.op.mk.intro.op.mk
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      a : ↑(F.obj A)
      isGalois✝² : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois B
      u : Quiver.Hom A X
      v : Quiver.Hom B X
      h : Eq (F.map u a) (F.map v b)
      Z : C
      z : ↑(F.obj Z)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois Z
      f : Quiver.Hom (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isGal …
      hf : Eq (F.map f (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isG …
      h✝ : True
      g : Quiver.Hom (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isGal …
      hg : Eq (F.map g (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isG …
      ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((((CategoryTheory.PreGal …
    -/
    refine ⟨⟨{ obj := Z, pt := z }⟩, ⟨f, hf⟩, ⟨g, hg⟩, ?_⟩
    /-
      case refine_2.intro.op.mk.intro.op.mk.intro.op.mk
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      a : ↑(F.obj A)
      isGalois✝² : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois B
      u : Quiver.Hom A X
      v : Quiver.Hom B X
      h : Eq (F.map u a) (F.map v b)
      Z : C
      z : ↑(F.obj Z)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois Z
      f : Quiver.Hom (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isGal …
      hf : Eq (F.map f (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isG …
      h✝ : True
      g : Quiver.Hom (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isGal …
      hg : Eq (F.map g (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isG …
      ⊢ Eq ((((CategoryTheory.PreGaloisCategory.PointedGaloisObject.incl F).op.comp  …
    -/
    apply evaluation_injective_of_isConnected F Z X z
    /-
      case refine_2.intro.op.mk.intro.op.mk.intro.op.mk.a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      a : ↑(F.obj A)
      isGalois✝² : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois B
      u : Quiver.Hom A X
      v : Quiver.Hom B X
      h : Eq (F.map u a) (F.map v b)
      Z : C
      z : ↑(F.obj Z)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois Z
      f : Quiver.Hom (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isGal …
      hf : Eq (F.map f (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isG …
      h✝ : True
      g : Quiver.Hom (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isGal …
      hg : Eq (F.map g (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isG …
      ⊢ Eq ((fun f => F.map f z) ((((CategoryTheory.PreGaloisCategory.PointedGaloisO …
    -/
    change F.map (f ≫ u) z = F.map (g ≫ v) z
    /-
      case refine_2.intro.op.mk.intro.op.mk.intro.op.mk.a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X A : C
      a : ↑(F.obj A)
      isGalois✝² : CategoryTheory.PreGaloisCategory.IsGalois A
      B : C
      b : ↑(F.obj B)
      isGalois✝¹ : CategoryTheory.PreGaloisCategory.IsGalois B
      u : Quiver.Hom A X
      v : Quiver.Hom B X
      h : Eq (F.map u a) (F.map v b)
      Z : C
      z : ↑(F.obj Z)
      isGalois✝ : CategoryTheory.PreGaloisCategory.IsGalois Z
      f : Quiver.Hom (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isGal …
      hf : Eq (F.map f (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isG …
      h✝ : True
      g : Quiver.Hom (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isGal …
      hg : Eq (F.map g (Opposite.unop { unop := { obj := Z, pt := z, isGalois := isG …
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f u) z) (F.map (CategoryTheory …
    -/
    rw [map_comp, FintypeCat.comp_apply, hf, map_comp, FintypeCat.comp_apply, hg, h]
    /-
      🎉 no goals
    -/


instance : HasColimit ((incl F).op ⋙ coyoneda) where
  exists_colimit := ⟨cocone F, isColimit F⟩


/-- The diagram sending each pointed Galois object to its automorphism group
as an object of `C`. -/
@[simps]
noncomputable def autGaloisSystem : PointedGaloisObject F ⥤ Grp.{u₂} where
  obj := fun A ↦ Grp.of <| Aut (A : C)
  map := fun {A B} f ↦ (autMapHom f : Aut (A : C) →* Aut (B : C))
  map_id := fun A ↦ by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
      inst✝ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      ⊢ Eq ({ obj := fun A => Grp.of (CategoryTheory.Aut A.obj), map := fun {A B} f  …
    -/
    ext (σ : Aut A.obj)
    /-
      case w
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
      inst✝ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      σ : CategoryTheory.Aut A.obj
      ⊢ Eq (({ obj := fun A => Grp.of (CategoryTheory.Aut A.obj), map := fun {A B} f …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_comp {A B C} f g := by
    /-
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{u₂, u₁} C✝
      inst✝ : CategoryTheory.GaloisCategory C✝
      F : CategoryTheory.Functor C✝ FintypeCat
      A B C : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      f : Quiver.Hom A B
      g : Quiver.Hom B C
      ⊢ Eq ({ obj := fun A => Grp.of (CategoryTheory.Aut A.obj), map := fun {A B} f  …
    -/
    ext (σ : Aut A.obj)
    /-
      case w
      C✝ : Type u₁
      inst✝¹ : CategoryTheory.Category.{u₂, u₁} C✝
      inst✝ : CategoryTheory.GaloisCategory C✝
      F : CategoryTheory.Functor C✝ FintypeCat
      A B C : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
      f : Quiver.Hom A B
      g : Quiver.Hom B C
      σ : CategoryTheory.Aut A.obj
      ⊢ Eq (({ obj := fun A => Grp.of (CategoryTheory.Aut A.obj), map := fun {A B} f …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The limit of `autGaloisSystem`. -/
noncomputable def AutGalois : Type (max u₁ u₂) :=
  (autGaloisSystem F ⋙ forget _).sections


noncomputable instance : Group (AutGalois F) :=
  inferInstanceAs <| Group (autGaloisSystem F ⋙ forget _).sections


/-- The canonical projection from `AutGalois F` to the `C`-automorphism group of each
pointed Galois object. -/
noncomputable def AutGalois.π (A : PointedGaloisObject F) : AutGalois F →* Aut (A : C) :=
  Grp.sectionsπMonoidHom (autGaloisSystem F) A

/- Not a `simp` lemma, because we usually don't want to expose the internals here. -/

lemma AutGalois.π_apply (A : PointedGaloisObject F) (x : AutGalois F) :
    AutGalois.π F A x = x.val A :=
  rfl


lemma autGaloisSystem_map_surjective ⦃A B : PointedGaloisObject F⦄ (f : A ⟶ B) :
    Function.Surjective ((autGaloisSystem F).map f) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    f : Quiver.Hom A B
    ⊢ Function.Surjective ⇑((CategoryTheory.PreGaloisCategory.autGaloisSystem F).m …
  -/
  intro (φ : Aut B.obj)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    f : Quiver.Hom A B
    φ : CategoryTheory.Aut B.obj
    ⊢ Exists fun a => Eq (((CategoryTheory.PreGaloisCategory.autGaloisSystem F).ma …
  -/
  obtain ⟨ψ, hψ⟩ := autMap_surjective_of_isGalois f.val φ
  /-
    case intro
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    f : Quiver.Hom A B
    φ : CategoryTheory.Aut B.obj
    ψ : CategoryTheory.Aut A.obj
    hψ : Eq (CategoryTheory.PreGaloisCategory.autMap f.val ψ) φ
    ⊢ Exists fun a => Eq (((CategoryTheory.PreGaloisCategory.autGaloisSystem F).ma …
  -/
  use ψ
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    f : Quiver.Hom A B
    φ : CategoryTheory.Aut B.obj
    ψ : CategoryTheory.Aut A.obj
    hψ : Eq (CategoryTheory.PreGaloisCategory.autMap f.val ψ) φ
    ⊢ Eq (((CategoryTheory.PreGaloisCategory.autGaloisSystem F).map f) ψ) φ
  -/
  simp only [autGaloisSystem_map]
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    f : Quiver.Hom A B
    φ : CategoryTheory.Aut B.obj
    ψ : CategoryTheory.Aut A.obj
    hψ : Eq (CategoryTheory.PreGaloisCategory.autMap f.val ψ) φ
    ⊢ Eq ((CategoryTheory.PreGaloisCategory.autMapHom f.val) ψ) φ
  -/
  exact hψ
  /-
    🎉 no goals
  -/


/-- Equality of elements of `AutGalois F` can be checked on the projections on each pointed
Galois object. -/
lemma AutGalois.ext {f g : AutGalois F}
    (h : ∀ (A : PointedGaloisObject F), AutGalois.π F A f = AutGalois.π F A g) : f = g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    f g : CategoryTheory.PreGaloisCategory.AutGalois F
    h : ∀ (A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F), Eq ((Categ …
    ⊢ Eq f g
  -/
  dsimp only [AutGalois]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    f g : CategoryTheory.PreGaloisCategory.AutGalois F
    h : ∀ (A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F), Eq ((Categ …
    ⊢ Eq f g
  -/
  ext A
  /-
    case a.h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{u₂, u₁} C
    inst✝ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    f g : CategoryTheory.PreGaloisCategory.AutGalois F
    h : ∀ (A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F), Eq ((Categ …
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (↑f A) (↑g A)
  -/
  exact h A
  /-
    🎉 no goals
  -/


/-- `autGalois.π` is surjective for every pointed Galois object. -/
theorem AutGalois.π_surjective (A : PointedGaloisObject F) :
    Function.Surjective (AutGalois.π F A) := fun (σ : Aut A.obj) ↦ by
  have (i : PointedGaloisObject F) : Finite ((autGaloisSystem F ⋙ forget _).obj i) :=
    inferInstanceAs <| Finite (Aut (i.obj))
  exact eval_section_surjective_of_surjective
    (autGaloisSystem F ⋙ forget _) (autGaloisSystem_map_surjective F) A σ


local notation "F'" => F ⋙ FintypeCat.incl


/-- The endomorphisms of `F` are isomorphic to the limit over the fibers of `F` on all
Galois objects. -/
noncomputable def endEquivSectionsFibers : End F ≃ (incl F ⋙ F').sections :=
  let i1 : End F ≃ End F' :=
    (FullyFaithful.whiskeringRight (FullyFaithful.ofFullyFaithful FintypeCat.incl) C).homEquiv
  let i2 : End F' ≅ (colimit ((incl F).op ⋙ coyoneda) ⟶ F') :=
    (yoneda.obj (F ⋙ FintypeCat.incl)).mapIso (colimit.isoColimitCocone ⟨cocone F, isColimit F⟩).op
  let i3 : (colimit ((incl F).op ⋙ coyoneda) ⟶ F') ≅ limit ((incl F ⋙ F') ⋙ uliftFunctor.{u₁}) :=
    colimitCoyonedaHomIsoLimit' (incl F) F'
  let i4 : limit (incl F ⋙ F' ⋙ uliftFunctor.{u₁}) ≃ ((incl F ⋙ F') ⋙ uliftFunctor.{u₁}).sections :=
    Types.limitEquivSections (incl F ⋙ (F ⋙ FintypeCat.incl) ⋙ uliftFunctor.{u₁, u₂})
  let i5 : ((incl F ⋙ F') ⋙ uliftFunctor.{u₁}).sections ≃ (incl F ⋙ F').sections :=
    (Types.sectionsEquiv (incl F ⋙ F')).symm
  i1.trans <| i2.toEquiv.trans <| i3.toEquiv.trans <| i4.trans i5


@[simp]
lemma endEquivSectionsFibers_π (f : End F) (A : PointedGaloisObject F) :
    (endEquivSectionsFibers F f).val A = f.app A A.pt := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (↑((CategoryTheory.PreGaloisCategory.endEquivSectionsFibers F) f) A) (f.a …
  -/
  dsimp [endEquivSectionsFibers, Types.sectionsEquiv]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (↑((CategoryTheory.Limits.Types.limitEquivSections ((CategoryTheory.PreGa …
  -/
  erw [Types.limitEquivSections_apply]
  simp only [colimitCoyonedaHomIsoLimit'_π_apply, incl_obj, comp_obj, FintypeCat.incl_obj, op_obj,
    FunctorToTypes.comp]
  change (((FullyFaithful.whiskeringRight (FullyFaithful.ofFullyFaithful
      FintypeCat.incl) C).homEquiv) f).app A
    (((colimit.ι _ _) ≫ (colimit.isoColimitCocone ⟨cocone F, isColimit F⟩).hom).app
      A _) = f.app A A.pt
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq ((((CategoryTheory.Functor.FullyFaithful.ofFullyFaithful FintypeCat.incl) …
  -/
  simp
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq ((((CategoryTheory.Functor.FullyFaithful.ofFullyFaithful FintypeCat.incl) …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Functorial isomorphism `Aut A ≅ F.obj A` for Galois objects `A`. -/
noncomputable def autIsoFibers :
    autGaloisSystem F ⋙ forget Grp ≅ incl F ⋙ F' :=
  NatIso.ofComponents (fun A ↦ ((evaluationEquivOfIsGalois F A A.pt).toIso))
    (fun {A B} f ↦ by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        inst✝¹ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
        f : Quiver.Hom A B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.PreGaloisCategory.a …
      -/
      ext (φ : Aut A.obj)
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        inst✝¹ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
        f : Quiver.Hom A B
        φ : CategoryTheory.Aut A.obj
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.PreGaloisCategory.a …
      -/
      dsimp
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        inst✝¹ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
        f : Quiver.Hom A B
        φ : CategoryTheory.Aut A.obj
        ⊢ Eq ((CategoryTheory.PreGaloisCategory.evaluationEquivOfIsGalois F B.obj B.pt …
      -/
      erw [evaluationEquivOfIsGalois_apply, evaluationEquivOfIsGalois_apply]
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{u₂, u₁} C
        inst✝¹ : CategoryTheory.GaloisCategory C
        F : CategoryTheory.Functor C FintypeCat
        inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
        A B : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
        f : Quiver.Hom A B
        φ : CategoryTheory.Aut A.obj
        ⊢ Eq (F.map (CategoryTheory.PreGaloisCategory.autMap f.val φ).hom B.pt) (F.map …
      -/
      simp [-Hom.comp, ← f.comp])
      /-
        🎉 no goals
      -/


lemma autIsoFibers_inv_app (A : PointedGaloisObject F) (b : F.obj A) :
    (autIsoFibers F).inv.app A b = (evaluationEquivOfIsGalois F A A.pt).symm b :=
  rfl


/-- The equivalence between endomorphisms of `F` and the limit over the automorphism groups
of all Galois objects. -/
noncomputable def endEquivAutGalois : End F ≃ AutGalois F :=
  let e1 := endEquivSectionsFibers F
  let e2 := ((Functor.sectionsFunctor _).mapIso (autIsoFibers F).symm).toEquiv
  e1.trans e2


lemma endEquivAutGalois_π (f : End F) (A : PointedGaloisObject F) :
    F.map (AutGalois.π F A (endEquivAutGalois F f)).hom A.pt = f.app A A.pt := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (F.map ((CategoryTheory.PreGaloisCategory.AutGalois.π F A) ((CategoryTheo …
  -/
  dsimp [endEquivAutGalois, AutGalois.π_apply]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (F.map (↑(((CategoryTheory.Functor.sectionsFunctor (CategoryTheory.PreGal …
  -/
  change F.map ((((sectionsFunctor _).map (autIsoFibers F).inv) _).val A).hom A.pt = _
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (F.map (↑((CategoryTheory.Functor.sectionsFunctor (CategoryTheory.PreGalo …
  -/
  dsimp [autIsoFibers]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (F.map ((CategoryTheory.PreGaloisCategory.evaluationEquivOfIsGalois F A.o …
  -/
  simp only [endEquivSectionsFibers_π]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (F.map ((CategoryTheory.PreGaloisCategory.evaluationEquivOfIsGalois F A.o …
  -/
  erw [evaluationEquivOfIsGalois_symm_fiber]
  /-
    🎉 no goals
  -/


@[simp]
theorem endEquivAutGalois_mul (f g : End F) :
    (endEquivAutGalois F) (g ≫ f) = (endEquivAutGalois F g) * (endEquivAutGalois F f) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f g : CategoryTheory.End F
    ⊢ Eq ((CategoryTheory.PreGaloisCategory.endEquivAutGalois F) (CategoryTheory.C …
  -/
  refine AutGalois.ext F (fun A ↦ evaluation_aut_injective_of_isConnected F A A.pt ?_)
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f g : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq ((fun f => F.map f.hom A.pt) ((CategoryTheory.PreGaloisCategory.AutGalois …
  -/
  simp only [map_mul, endEquivAutGalois_π, Aut.Aut_mul_def, NatTrans.comp_app, Iso.trans_hom]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f g : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.app A.obj) (f.app A.obj) A.pt) (F. …
  -/
  simp only [map_comp, FintypeCat.comp_apply, endEquivAutGalois_π]
  change f.app A (g.app A A.pt) =
    (f.app A ≫ F.map ((AutGalois.π F A) ((endEquivAutGalois F) g)).hom) A.pt
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f g : CategoryTheory.End F
    A : CategoryTheory.PreGaloisCategory.PointedGaloisObject F
    ⊢ Eq (f.app A.obj (g.app A.obj A.pt)) (CategoryTheory.CategoryStruct.comp (f.a …
  -/
  rw [← f.naturality, FintypeCat.comp_apply, endEquivAutGalois_π]
  /-
    🎉 no goals
  -/


/-- The monoid isomorphism between endomorphisms of `F` and the (multiplicative opposite of the)
limit of automorphism groups of all Galois objects. -/
noncomputable def endMulEquivAutGalois : End F ≃* (AutGalois F)ᵐᵒᵖ :=
                                                                          /-
                                                                            C : Type u₁
                                                                            inst✝² : CategoryTheory.Category.{u₂, u₁} C
                                                                            inst✝¹ : CategoryTheory.GaloisCategory C
                                                                            F : CategoryTheory.Functor C FintypeCat
                                                                            inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
                                                                            ⊢ ∀ (x y : CategoryTheory.End F), Eq (((CategoryTheory.PreGaloisCategory.endEq …
                                                                          -/
  MulEquiv.mk (Equiv.trans (endEquivAutGalois F) MulOpposite.opEquiv) (by simp)
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma endMulEquivAutGalois_pi (f : End F) (A : PointedGaloisObject F) :
    F.map (AutGalois.π F A (endMulEquivAutGalois F f).unop).hom A.2 = f.app A A.pt :=
  endEquivAutGalois_π F f A


/-- Any endomorphism of a fiber functor is a unit. -/
theorem FibreFunctor.end_isUnit (f : End F) : IsUnit f :=
  (isUnit_map_iff (endMulEquivAutGalois F) _).mp
    (Group.isUnit ((endMulEquivAutGalois F) f))


/-- Any endomorphism of a fiber functor is an isomorphism. -/
instance FibreFunctor.end_isIso (f : End F) : IsIso f := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    ⊢ CategoryTheory.IsIso f
  -/
  rw [← isUnit_iff_isIso]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{u₂, u₁} C
    inst✝¹ : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.End F
    ⊢ IsUnit f
  -/
  exact FibreFunctor.end_isUnit F f
  /-
    🎉 no goals
  -/


/-- The automorphism group of `F` is multiplicatively isomorphic to
(the multiplicative opposite of) the limit over the automorphism groups of
the Galois objects. -/
noncomputable def autMulEquivAutGalois : Aut F ≃* (AutGalois F)ᵐᵒᵖ where
  toFun := MonoidHom.comp (endMulEquivAutGalois F) (Aut.toEnd F)
  invFun t := asIso ((endMulEquivAutGalois F).symm t)
  left_inv t := by
    simp only [MonoidHom.coe_comp, MonoidHom.coe_coe, Function.comp_apply,
      MulEquiv.symm_apply_apply]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      t : CategoryTheory.Aut F
      ⊢ Eq (CategoryTheory.asIso ((CategoryTheory.Aut.toEnd F) t)) t
    -/
    exact Aut.ext rfl
    /-
      🎉 no goals
    -/
  right_inv t := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      t : MulOpposite (CategoryTheory.PreGaloisCategory.AutGalois F)
      ⊢ Eq (((↑(CategoryTheory.PreGaloisCategory.endMulEquivAutGalois F)).comp (Cate …
    -/
    simp only [MonoidHom.coe_comp, MonoidHom.coe_coe, Function.comp_apply, Aut.toEnd_apply]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{u₂, u₁} C
      inst✝¹ : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      t : MulOpposite (CategoryTheory.PreGaloisCategory.AutGalois F)
      ⊢ Eq ((CategoryTheory.PreGaloisCategory.endMulEquivAutGalois F) ↑((CategoryThe …
    -/
    exact (MulEquiv.eq_symm_apply (endMulEquivAutGalois F)).mp rfl
    /-
      🎉 no goals
    -/
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{u₂, u₁} C
                   inst✝¹ : CategoryTheory.GaloisCategory C
                   F : CategoryTheory.Functor C FintypeCat
                   inst✝ : CategoryTheory.PreGaloisCategory.FiberFunctor F
                   ⊢ ∀ (x y : CategoryTheory.Aut F), Eq ({ toFun := ⇑((↑(CategoryTheory.PreGalois …
                 -/
  map_mul' := by simp [map_mul]
                 /-
                   🎉 no goals
                 -/


lemma autMulEquivAutGalois_π (f : Aut F) (A : C) [IsGalois A] (a : F.obj A) :
    F.map (AutGalois.π F { obj := A, pt := a } (autMulEquivAutGalois F f).unop).hom a =
      f.hom.app A a := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.Aut F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    ⊢ Eq (F.map ((CategoryTheory.PreGaloisCategory.AutGalois.π F { obj := A, pt := …
  -/
  dsimp [autMulEquivAutGalois, endMulEquivAutGalois]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.Aut F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    ⊢ Eq (F.map ((CategoryTheory.PreGaloisCategory.AutGalois.π F { obj := A, pt := …
  -/
  rw [endEquivAutGalois_π]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    f : CategoryTheory.Aut F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    ⊢ Eq ((↑((CategoryTheory.Aut.unitsEndEquivAut F).symm f)).app { obj := A, pt : …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma autMulEquivAutGalois_symm_app (x : AutGalois F) (A : C) [IsGalois A] (a : F.obj A) :
    ((autMulEquivAutGalois F).symm ⟨x⟩).hom.app A a =
      F.map (AutGalois.π F ⟨A, a, inferInstance⟩ x).hom a := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    x : CategoryTheory.PreGaloisCategory.AutGalois F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    ⊢ Eq (((CategoryTheory.PreGaloisCategory.autMulEquivAutGalois F).symm { unop'  …
  -/
  rw [← autMulEquivAutGalois_π, MulEquiv.apply_symm_apply]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    x : CategoryTheory.PreGaloisCategory.AutGalois F
    A : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois A
    a : ↑(F.obj A)
    ⊢ Eq (F.map ((CategoryTheory.PreGaloisCategory.AutGalois.π F { obj := A, pt := …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The `Aut F` action on the fiber of a Galois object is transitive. See
`pretransitive_of_isConnected` for the same result for connected objects. -/
theorem FiberFunctor.isPretransitive_of_isGalois (X : C) [IsGalois X] :
    MulAction.IsPretransitive (Aut F) (F.obj X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    ⊢ MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj X)
  -/
  refine ⟨fun x y ↦ ?_⟩
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    x y : ↑(F.obj X)
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  obtain ⟨(φ : Aut X), h⟩ := MulAction.IsPretransitive.exists_smul_eq (M := Aut X) x y
  /-
    case intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    x y : ↑(F.obj X)
    φ : CategoryTheory.Aut X
    h : Eq (HSMul.hSMul φ x) y
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  obtain ⟨a, ha⟩ := AutGalois.π_surjective F ⟨X, x, inferInstance⟩ φ
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    x y : ↑(F.obj X)
    φ : CategoryTheory.Aut X
    h : Eq (HSMul.hSMul φ x) y
    a : CategoryTheory.PreGaloisCategory.AutGalois F
    ha : Eq ((CategoryTheory.PreGaloisCategory.AutGalois.π F { obj := X, pt := x,  …
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  use (autMulEquivAutGalois F).symm ⟨a⟩
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsGalois X
    x y : ↑(F.obj X)
    φ : CategoryTheory.Aut X
    h : Eq (HSMul.hSMul φ x) y
    a : CategoryTheory.PreGaloisCategory.AutGalois F
    ha : Eq ((CategoryTheory.PreGaloisCategory.AutGalois.π F { obj := X, pt := x,  …
    ⊢ Eq (HSMul.hSMul ((CategoryTheory.PreGaloisCategory.autMulEquivAutGalois F).s …
  -/
  simpa [mulAction_def, ha]
  /-
    🎉 no goals
  -/


/-- The `Aut F` action on the fiber of a connected object is transitive. For a version
with less restrictive universe assumptions, see `FiberFunctor.isPretransitive_of_isConnected`. -/
private instance FiberFunctor.isPretransitive_of_isConnected' (X : C) [IsConnected X] :
    MulAction.IsPretransitive (Aut F) (F.obj X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    ⊢ MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj X)
  -/
  obtain ⟨A, f, hgal⟩ := exists_hom_from_galois_of_connected F X
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    ⊢ MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj X)
  -/
  have hs : Function.Surjective (F.map f) := surjective_of_nonempty_fiber_of_isConnected F f
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    ⊢ MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj X)
  -/
  refine ⟨fun x y ↦ ?_⟩
  /-
    case intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    x y : ↑(F.obj X)
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  obtain ⟨a, ha⟩ := hs x
  /-
    case intro.intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    x y : ↑(F.obj X)
    a : ↑(F.obj A)
    ha : Eq (F.map f a) x
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  obtain ⟨b, hb⟩ := hs y
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    x y : ↑(F.obj X)
    a : ↑(F.obj A)
    ha : Eq (F.map f a) x
    b : ↑(F.obj A)
    hb : Eq (F.map f b) y
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  have : MulAction.IsPretransitive (Aut F) (F.obj A) := isPretransitive_of_isGalois F A
  /-
    case intro.intro.intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    x y : ↑(F.obj X)
    a : ↑(F.obj A)
    ha : Eq (F.map f a) x
    b : ↑(F.obj A)
    hb : Eq (F.map f b) y
    this : MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj A)
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  obtain ⟨σ, (hσ : σ.hom.app A a = b)⟩ := MulAction.exists_smul_eq (Aut F) a b
  /-
    case intro.intro.intro.intro.intro
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    x y : ↑(F.obj X)
    a : ↑(F.obj A)
    ha : Eq (F.map f a) x
    b : ↑(F.obj A)
    hb : Eq (F.map f b) y
    this : MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj A)
    σ : CategoryTheory.Aut F
    hσ : Eq (σ.hom.app A a) b
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  use σ
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    x y : ↑(F.obj X)
    a : ↑(F.obj A)
    ha : Eq (F.map f a) x
    b : ↑(F.obj A)
    hb : Eq (F.map f b) y
    this : MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj A)
    σ : CategoryTheory.Aut F
    hσ : Eq (σ.hom.app A a) b
    ⊢ Eq (HSMul.hSMul σ x) y
  -/
  rw [← ha, ← hb]
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    x y : ↑(F.obj X)
    a : ↑(F.obj A)
    ha : Eq (F.map f a) x
    b : ↑(F.obj A)
    hb : Eq (F.map f b) y
    this : MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj A)
    σ : CategoryTheory.Aut F
    hσ : Eq (σ.hom.app A a) b
    ⊢ Eq (HSMul.hSMul σ (F.map f a)) (F.map f b)
  -/
  show (F.map f ≫ σ.hom.app X) a = F.map f b
  /-
    case h
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{u₂, u₁} C
    inst✝² : CategoryTheory.GaloisCategory C
    F : CategoryTheory.Functor C FintypeCat
    inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
    X : C
    inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
    A : C
    f : Quiver.Hom A X
    hgal : CategoryTheory.PreGaloisCategory.IsGalois A
    hs : Function.Surjective (F.map f)
    x y : ↑(F.obj X)
    a : ↑(F.obj A)
    ha : Eq (F.map f a) x
    b : ↑(F.obj A)
    hb : Eq (F.map f b) y
    this : MulAction.IsPretransitive (CategoryTheory.Aut F) ↑(F.obj A)
    σ : CategoryTheory.Aut F
    hσ : Eq (σ.hom.app A a) b
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (σ.hom.app X) a) (F.map f b)
  -/
  rw [σ.hom.naturality, FintypeCat.comp_apply, hσ]
  /-
    🎉 no goals
  -/


/-- The `Aut F` action on the fiber of a connected object is transitive. -/
instance FiberFunctor.isPretransitive_of_isConnected (X : C) [IsConnected X] :
    MulAction.IsPretransitive (Aut F) (F.obj X) where
  exists_smul_eq x y := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    let F' : C ⥤ FintypeCat.{u₂} := F ⋙ FintypeCat.uSwitch.{w, u₂}
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      F' : CategoryTheory.Functor C FintypeCat := F.comp FintypeCat.uSwitch
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    letI : FiberFunctor F' := FiberFunctor.comp_right _
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      F' : CategoryTheory.Functor C FintypeCat := F.comp FintypeCat.uSwitch
      this : CategoryTheory.PreGaloisCategory.FiberFunctor F' := CategoryTheory.PreG …
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    let e (Y : C) : F'.obj Y ≃ F.obj Y := (F.obj Y).uSwitchEquiv
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      F' : CategoryTheory.Functor C FintypeCat := F.comp FintypeCat.uSwitch
      this : CategoryTheory.PreGaloisCategory.FiberFunctor F' := CategoryTheory.PreG …
      e : (Y : C) → Equiv ↑(F'.obj Y) ↑(F.obj Y) := fun Y => (F.obj Y).uSwitchEquiv
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    set x' : F'.obj X := (e X).symm x with hx'
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      F' : CategoryTheory.Functor C FintypeCat := F.comp FintypeCat.uSwitch
      this : CategoryTheory.PreGaloisCategory.FiberFunctor F' := CategoryTheory.PreG …
      e : (Y : C) → Equiv ↑(F'.obj Y) ↑(F.obj Y) := fun Y => (F.obj Y).uSwitchEquiv
      x' : ↑(F'.obj X) := (e X).symm x
      hx' : Eq x' ((e X).symm x)
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    set y' : F'.obj X := (e X).symm y with hy'
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      F' : CategoryTheory.Functor C FintypeCat := F.comp FintypeCat.uSwitch
      this : CategoryTheory.PreGaloisCategory.FiberFunctor F' := CategoryTheory.PreG …
      e : (Y : C) → Equiv ↑(F'.obj Y) ↑(F.obj Y) := fun Y => (F.obj Y).uSwitchEquiv
      x' : ↑(F'.obj X) := (e X).symm x
      hx' : Eq x' ((e X).symm x)
      y' : ↑(F'.obj X) := (e X).symm y
      hy' : Eq y' ((e X).symm y)
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    obtain ⟨g', (hg' : g'.hom.app X x' = y')⟩ := MulAction.exists_smul_eq (Aut F') x' y'
    let gapp (Y : C) : F.obj Y ≅ F.obj Y := FintypeCat.equivEquivIso <|
      (e Y).symm.trans <| (FintypeCat.equivEquivIso.symm (g'.app Y)).trans (e Y)
    let g : F ≅ F := NatIso.ofComponents gapp <| fun {X Y} f ↦ by
      ext x
      simp only [FintypeCat.comp_apply, FintypeCat.equivEquivIso_apply_hom,
        Equiv.trans_apply, FintypeCat.equivEquivIso_symm_apply_apply, Iso.app_hom, gapp, e]
      erw [FintypeCat.uSwitchEquiv_naturality (F.map f)]
      rw [← Functor.comp_map, ← FunctorToFintypeCat.naturality]
      simp only [comp_obj, Functor.comp_map, F']
      rw [FintypeCat.uSwitchEquiv_symm_naturality (F.map f)]
    /-
      case intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      F' : CategoryTheory.Functor C FintypeCat := F.comp FintypeCat.uSwitch
      this : CategoryTheory.PreGaloisCategory.FiberFunctor F' := CategoryTheory.PreG …
      e : (Y : C) → Equiv ↑(F'.obj Y) ↑(F.obj Y) := fun Y => (F.obj Y).uSwitchEquiv
      x' : ↑(F'.obj X) := (e X).symm x
      hx' : Eq x' ((e X).symm x)
      y' : ↑(F'.obj X) := (e X).symm y
      hy' : Eq y' ((e X).symm y)
      g' : CategoryTheory.Aut F'
      hg' : Eq (g'.hom.app X x') y'
      gapp : (Y : C) → CategoryTheory.Iso (F.obj Y) (F.obj Y) := fun Y => FintypeCat …
      g : CategoryTheory.Iso F F := CategoryTheory.NatIso.ofComponents gapp ⋯
      ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
    -/
    refine ⟨g, show (gapp X).hom x = y from ?_⟩
    simp only [FintypeCat.equivEquivIso_apply_hom, Equiv.trans_apply,
      FintypeCat.equivEquivIso_symm_apply_apply, Iso.app_hom, gapp]
    /-
      case intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{u₂, u₁} C
      inst✝² : CategoryTheory.GaloisCategory C
      F : CategoryTheory.Functor C FintypeCat
      inst✝¹ : CategoryTheory.PreGaloisCategory.FiberFunctor F
      X : C
      inst✝ : CategoryTheory.PreGaloisCategory.IsConnected X
      x y : ↑(F.obj X)
      F' : CategoryTheory.Functor C FintypeCat := F.comp FintypeCat.uSwitch
      this : CategoryTheory.PreGaloisCategory.FiberFunctor F' := CategoryTheory.PreG …
      e : (Y : C) → Equiv ↑(F'.obj Y) ↑(F.obj Y) := fun Y => (F.obj Y).uSwitchEquiv
      x' : ↑(F'.obj X) := (e X).symm x
      hx' : Eq x' ((e X).symm x)
      y' : ↑(F'.obj X) := (e X).symm y
      hy' : Eq y' ((e X).symm y)
      g' : CategoryTheory.Aut F'
      hg' : Eq (g'.hom.app X x') y'
      gapp : (Y : C) → CategoryTheory.Iso (F.obj Y) (F.obj Y) := fun Y => FintypeCat …
      g : CategoryTheory.Iso F F := CategoryTheory.NatIso.ofComponents gapp ⋯
      ⊢ Eq ((e X) (g'.hom.app X ((e X).symm x))) y
    -/
    rw [← hx', hg', hy', Equiv.apply_symm_apply]
    /-
      🎉 no goals
    -/


