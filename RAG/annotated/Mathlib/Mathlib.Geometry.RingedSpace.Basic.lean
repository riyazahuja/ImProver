/-- The type of Ringed spaces, as an abbreviation for `SheafedSpace CommRingCat`. -/
abbrev RingedSpace : TypeMax.{u+1, v+1} :=
  SheafedSpace.{v+1, v, u} CommRingCat.{v}


@[simp]
lemma res_zero {X : RingedSpace.{u}} {U V : TopologicalSpace.Opens X}
    (hUV : U ≤ V) : (0 : X.presheaf.obj (op V)) |_ U =
      (0 : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj (op U))) :=
  RingHom.map_zero _


instance : CoeSort RingedSpace Type* where
  coe X := X.carrier


/-- If the germ of a section `f` is zero in the stalk at `x`, then `f` is zero on some neighbourhood
around `x`. -/
lemma exists_res_eq_zero_of_germ_eq_zero (U : Opens X) (f : X.presheaf.obj (op U)) (x : U)
    (h : X.presheaf.germ U x.val x.property f = 0) :
    ∃ (V : Opens X) (i : V ⟶ U) (_ : x.1 ∈ V), X.presheaf.map i.op f = 0 := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    h : Eq ((X.presheaf.germ U ↑x ⋯).hom f) 0
    ⊢ Exists fun V => Exists fun i => Exists fun x => Eq ((X.presheaf.map i.op).ho …
  -/
  have h1 : X.presheaf.germ U x.val x.property f = X.presheaf.germ U x.val x.property 0 := by simpa
  obtain ⟨V, hv, i, _, (hv4 : (X.presheaf.map i.op) f = (X.presheaf.map _) 0)⟩ :=
    TopCat.Presheaf.germ_eq X.presheaf x.1 x.2 x.2 f 0 h1
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    h : Eq ((X.presheaf.germ U ↑x ⋯).hom f) 0
    h1 : Eq ((X.presheaf.germ U ↑x ⋯).hom f) ((X.presheaf.germ U ↑x ⋯).hom 0)
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hv : Membership.mem V ↑x
    i w✝ : Quiver.Hom V U
    hv4 : Eq ((X.presheaf.map i.op).hom f) ((X.presheaf.map w✝.op).hom 0)
    ⊢ Exists fun V => Exists fun i => Exists fun x => Eq ((X.presheaf.map i.op).ho …
  -/
  use V, i, hv
  /-
    case h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : Subtype fun x => Membership.mem U x
    h : Eq ((X.presheaf.germ U ↑x ⋯).hom f) 0
    h1 : Eq ((X.presheaf.germ U ↑x ⋯).hom f) ((X.presheaf.germ U ↑x ⋯).hom 0)
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hv : Membership.mem V ↑x
    i w✝ : Quiver.Hom V U
    hv4 : Eq ((X.presheaf.map i.op).hom f) ((X.presheaf.map w✝.op).hom 0)
    ⊢ Eq ((X.presheaf.map i.op).hom f) 0
  -/
  simpa using hv4
  /-
    🎉 no goals
  -/


/--
If the germ of a section `f` is a unit in the stalk at `x`, then `f` must be a unit on some small
neighborhood around `x`.
-/
theorem isUnit_res_of_isUnit_germ (U : Opens X) (f : X.presheaf.obj (op U)) (x : X) (hx : x ∈ U)
    (h : IsUnit (X.presheaf.germ U x hx f)) :
    ∃ (V : Opens X) (i : V ⟶ U) (_ : x ∈ V), IsUnit (X.presheaf.map i.op f) := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    h : IsUnit ((X.presheaf.germ U x hx).hom f)
    ⊢ Exists fun V => Exists fun i => Exists fun x => IsUnit ((X.presheaf.map i.op …
  -/
  obtain ⟨g', heq⟩ := h.exists_right_inv
  /-
    case intro
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    h : IsUnit ((X.presheaf.germ U x hx).hom f)
    g' : ↑(X.presheaf.stalk x)
    heq : Eq (HMul.hMul ((X.presheaf.germ U x hx).hom f) g') 1
    ⊢ Exists fun V => Exists fun i => Exists fun x => IsUnit ((X.presheaf.map i.op …
  -/
  obtain ⟨V, hxV, g, rfl⟩ := X.presheaf.germ_exist x g'
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    h : IsUnit ((X.presheaf.germ U x hx).hom f)
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxV : Membership.mem V x
    g : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := V })
    heq : Eq (HMul.hMul ((X.presheaf.germ U x hx).hom f) ((X.presheaf.germ V x hxV …
    ⊢ Exists fun V => Exists fun i => Exists fun x => IsUnit ((X.presheaf.map i.op …
  -/
  let W := U ⊓ V
  /-
    case intro.intro.intro.intro
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    h : IsUnit ((X.presheaf.germ U x hx).hom f)
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxV : Membership.mem V x
    g : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := V })
    heq : Eq (HMul.hMul ((X.presheaf.germ U x hx).hom f) ((X.presheaf.germ V x hxV …
    W : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := Min.min U V
    ⊢ Exists fun V => Exists fun i => Exists fun x => IsUnit ((X.presheaf.map i.op …
  -/
  have hxW : x ∈ W := ⟨hx, hxV⟩
  -- Porting note: `erw` can't write into `HEq`, so this is replaced with another `HEq` in the
  -- desired form
  replace heq : (X.presheaf.germ _ x hxW) ((X.presheaf.map (U.infLELeft V).op) f *
      (X.presheaf.map (U.infLERight V).op) g) = (X.presheaf.germ _ x hxW) 1 := by
    dsimp [germ]
    erw [map_mul, map_one, show X.presheaf.germ _ x hxW ((X.presheaf.map (U.infLELeft V).op) f) =
      X.presheaf.germ U x hx f from X.presheaf.germ_res_apply (Opens.infLELeft U V) x hxW f,
      show X.presheaf.germ _ x hxW (X.presheaf.map (U.infLERight V).op g) =
      X.presheaf.germ _ x hxV g from X.presheaf.germ_res_apply (Opens.infLERight U V) x hxW g]
    exact heq
  -- note: we have to force lean to resynthesize this as <...>.hom _ = <...>.hom _
  obtain ⟨W', hxW', i₁, i₂, (heq' : (X.presheaf.map i₁.op) _ = (X.presheaf.map i₂.op) 1)⟩ :=
    X.presheaf.germ_eq x hxW hxW _ _ heq
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    h : IsUnit ((X.presheaf.germ U x hx).hom f)
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxV : Membership.mem V x
    g : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := V })
    W : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := Min.min U V
    hxW : Membership.mem W x
    heq : Eq ((X.presheaf.germ W x hxW).hom (HMul.hMul ((X.presheaf.map (U.infLELe …
    W' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW' : Membership.mem W' x
    i₁ i₂ : Quiver.Hom W' W
    heq' : Eq ((X.presheaf.map i₁.op).hom (HMul.hMul ((X.presheaf.map (U.infLELeft …
    ⊢ Exists fun V => Exists fun i => Exists fun x => IsUnit ((X.presheaf.map i.op …
  -/
  use W', i₁ ≫ Opens.infLELeft U V, hxW'
  /-
    case h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    h : IsUnit ((X.presheaf.germ U x hx).hom f)
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxV : Membership.mem V x
    g : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := V })
    W : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := Min.min U V
    hxW : Membership.mem W x
    heq : Eq ((X.presheaf.germ W x hxW).hom (HMul.hMul ((X.presheaf.map (U.infLELe …
    W' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW' : Membership.mem W' x
    i₁ i₂ : Quiver.Hom W' W
    heq' : Eq ((X.presheaf.map i₁.op).hom (HMul.hMul ((X.presheaf.map (U.infLELeft …
    ⊢ IsUnit ((X.presheaf.map (CategoryTheory.CategoryStruct.comp i₁ (U.infLELeft  …
  -/
  simp only [map_mul, map_one] at heq'
  /-
    case h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    h : IsUnit ((X.presheaf.germ U x hx).hom f)
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxV : Membership.mem V x
    g : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := V })
    W : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := Min.min U V
    hxW : Membership.mem W x
    heq : Eq ((X.presheaf.germ W x hxW).hom (HMul.hMul ((X.presheaf.map (U.infLELe …
    W' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxW' : Membership.mem W' x
    i₁ i₂ : Quiver.Hom W' W
    heq' : Eq (HMul.hMul ((X.presheaf.map i₁.op).hom ((X.presheaf.map (U.infLELeft …
    ⊢ IsUnit ((X.presheaf.map (CategoryTheory.CategoryStruct.comp i₁ (U.infLELeft  …
  -/
  simpa using isUnit_of_mul_eq_one _ _ heq'
  /-
    🎉 no goals
  -/


/-- Specialize `TopCat.Presheaf.germ_res_apply` to sheaves of rings.

This is unfortunately needed because the results on presheaves are stated using the
`ConcreteCategory.instFunLike` instance, which is not reducibly equal to the actual coercion of
morphisms in `CommRingCat` to functions.
-/
lemma _root_.CommRingCat.germ_res_apply
    {X : TopCat} (F : Presheaf CommRingCat X)
    {U V : Opens X} (i : U ⟶ V) (x : X) (hx : x ∈ U) (s) :
    F.germ U x hx (F.map i.op s) = F.germ V x (i.le hx) s :=
  F.germ_res_apply _ _ _ _


/-- Specialize `TopCat.Presheaf.germ_res_apply'` to sheaves of rings.

This is unfortunately needed because the results on presheaves are stated using the
`ConcreteCategory.instFunLike` instance, which is not reducibly equal to the actual coercion of
morphisms in `CommRingCat` to functions.
-/
lemma _root_.CommRingCat.germ_res_apply'
    {X : TopCat} (F : Presheaf CommRingCat X)
    {U V : Opens X} (i : op V ⟶ op U) (x : X) (hx : x ∈ U) (s) :
    F.germ U x hx (F.map i s) = F.germ V x (i.unop.le hx) s :=
  F.germ_res_apply' _ _ _ _


/-- If a section `f` is a unit in each stalk, `f` must be a unit. -/
theorem isUnit_of_isUnit_germ (U : Opens X) (f : X.presheaf.obj (op U))
    (h : ∀ (x) (hx : x ∈ U), IsUnit (X.presheaf.germ U x hx f)) : IsUnit f := by
  -- We pick a cover of `U` by open sets `V x`, such that `f` is a unit on each `V x`.
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    h : ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), IsUnit ((X.preshe …
    ⊢ IsUnit f
  -/
  choose V iVU m h_unit using fun x : U => X.isUnit_res_of_isUnit_germ U f x x.2 (h x.1 x.2)
  have hcover : U ≤ iSup V := by
    intro x hxU
    -- Porting note: in Lean3 `rw` is sufficient
    erw [Opens.mem_iSup]
    exact ⟨⟨x, hxU⟩, m ⟨x, hxU⟩⟩
  -- Let `g x` denote the inverse of `f` in `U x`.
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    h : ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), IsUnit ((X.preshe …
    V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑↑X.toPresh …
    iVU : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
    m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
    h_unit : ∀ (x : Subtype fun x => Membership.mem U x), IsUnit ((X.presheaf.map  …
    hcover : LE.le U (iSup V)
    ⊢ IsUnit f
  -/
  choose g hg using fun x : U => IsUnit.exists_right_inv (h_unit x)
  have ic : IsCompatible (sheaf X).val V g := by
    intro x y
    apply section_ext X.sheaf (V x ⊓ V y)
    rintro z ⟨hzVx, hzVy⟩
    rw [germ_res_apply, germ_res_apply]
    apply (h z ((iVU x).le hzVx)).mul_right_inj.mp
    -- Porting note: now need explicitly typing the rewrites
    -- note: this is bad, I think we should replace the `FunLike` on
    -- concrete category with `CoeFun`
    rw [← CommRingCat.germ_res_apply X.presheaf (iVU x) z hzVx f]
    -- Porting note: change was not necessary in Lean3
    change X.presheaf.germ _ z hzVx _ * (X.presheaf.germ _ z hzVx _) =
      X.presheaf.germ _ z hzVx _ * X.presheaf.germ _ z hzVy (g y)
    rw [← RingHom.map_mul,
      congr_arg (X.presheaf.germ (V x) z hzVx) (hg x),
      CommRingCat.germ_res_apply X.presheaf _ _ _ f,
      ← CommRingCat.germ_res_apply X.presheaf (iVU y) z hzVy f,
      ← RingHom.map_mul,
      congr_arg (X.presheaf.germ (V y) z hzVy) (hg y), RingHom.map_one, RingHom.map_one]
  -- We claim that these local inverses glue together to a global inverse of `f`.
  obtain ⟨gl, gl_spec, -⟩ :
    -- We need to rephrase the result from `ConcreteCategory` to `CommRingCat`.
    ∃ gl : X.presheaf.obj (op U), (∀ i, ((sheaf X).val.map (iVU i).op) gl = g i) ∧ _ :=
    X.sheaf.existsUnique_gluing' V U iVU hcover g ic
  /-
    case intro.intro
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    h : ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), IsUnit ((X.preshe …
    V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑↑X.toPresh …
    iVU : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
    m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
    h_unit : ∀ (x : Subtype fun x => Membership.mem U x), IsUnit ((X.presheaf.map  …
    hcover : LE.le U (iSup V)
    g : (x : Subtype fun x => Membership.mem U x) → ↑(X.presheaf.obj { unop := V x …
    hg : ∀ (x : Subtype fun x => Membership.mem U x), Eq (HMul.hMul ((X.presheaf.m …
    ic : TopCat.Presheaf.IsCompatible (AlgebraicGeometry.SheafedSpace.sheaf X).val …
    gl : ↑(X.presheaf.obj { unop := U })
    gl_spec : ∀ (i : Subtype fun x => Membership.mem U x), Eq (((AlgebraicGeometry …
    ⊢ IsUnit f
  -/
  apply isUnit_of_mul_eq_one f gl
  /-
    case intro.intro
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    h : ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), IsUnit ((X.preshe …
    V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑↑X.toPresh …
    iVU : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
    m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
    h_unit : ∀ (x : Subtype fun x => Membership.mem U x), IsUnit ((X.presheaf.map  …
    hcover : LE.le U (iSup V)
    g : (x : Subtype fun x => Membership.mem U x) → ↑(X.presheaf.obj { unop := V x …
    hg : ∀ (x : Subtype fun x => Membership.mem U x), Eq (HMul.hMul ((X.presheaf.m …
    ic : TopCat.Presheaf.IsCompatible (AlgebraicGeometry.SheafedSpace.sheaf X).val …
    gl : ↑(X.presheaf.obj { unop := U })
    gl_spec : ∀ (i : Subtype fun x => Membership.mem U x), Eq (((AlgebraicGeometry …
    ⊢ Eq (HMul.hMul f gl) 1
  -/
  apply X.sheaf.eq_of_locally_eq' V U iVU hcover
  /-
    case intro.intro.h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    h : ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), IsUnit ((X.preshe …
    V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑↑X.toPresh …
    iVU : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
    m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
    h_unit : ∀ (x : Subtype fun x => Membership.mem U x), IsUnit ((X.presheaf.map  …
    hcover : LE.le U (iSup V)
    g : (x : Subtype fun x => Membership.mem U x) → ↑(X.presheaf.obj { unop := V x …
    hg : ∀ (x : Subtype fun x => Membership.mem U x), Eq (HMul.hMul ((X.presheaf.m …
    ic : TopCat.Presheaf.IsCompatible (AlgebraicGeometry.SheafedSpace.sheaf X).val …
    gl : ↑(X.presheaf.obj { unop := U })
    gl_spec : ∀ (i : Subtype fun x => Membership.mem U x), Eq (((AlgebraicGeometry …
    ⊢ ∀ (i : Subtype fun x => Membership.mem U x), Eq (((AlgebraicGeometry.Sheafed …
  -/
  intro i
  -- We need to rephrase the goal from `ConcreteCategory` to `CommRingCat`.
  /-
    case intro.intro.h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    h : ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), IsUnit ((X.preshe …
    V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑↑X.toPresh …
    iVU : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
    m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
    h_unit : ∀ (x : Subtype fun x => Membership.mem U x), IsUnit ((X.presheaf.map  …
    hcover : LE.le U (iSup V)
    g : (x : Subtype fun x => Membership.mem U x) → ↑(X.presheaf.obj { unop := V x …
    hg : ∀ (x : Subtype fun x => Membership.mem U x), Eq (HMul.hMul ((X.presheaf.m …
    ic : TopCat.Presheaf.IsCompatible (AlgebraicGeometry.SheafedSpace.sheaf X).val …
    gl : ↑(X.presheaf.obj { unop := U })
    gl_spec : ∀ (i : Subtype fun x => Membership.mem U x), Eq (((AlgebraicGeometry …
    i : Subtype fun x => Membership.mem U x
    ⊢ Eq (((AlgebraicGeometry.SheafedSpace.sheaf X).val.map (iVU i).op) (HMul.hMul …
  -/
  show ((sheaf X).val.map (iVU i).op).hom (f * gl) = ((sheaf X).val.map (iVU i).op) 1
  /-
    case intro.intro.h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    h : ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), IsUnit ((X.preshe …
    V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑↑X.toPresh …
    iVU : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
    m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
    h_unit : ∀ (x : Subtype fun x => Membership.mem U x), IsUnit ((X.presheaf.map  …
    hcover : LE.le U (iSup V)
    g : (x : Subtype fun x => Membership.mem U x) → ↑(X.presheaf.obj { unop := V x …
    hg : ∀ (x : Subtype fun x => Membership.mem U x), Eq (HMul.hMul ((X.presheaf.m …
    ic : TopCat.Presheaf.IsCompatible (AlgebraicGeometry.SheafedSpace.sheaf X).val …
    gl : ↑(X.presheaf.obj { unop := U })
    gl_spec : ∀ (i : Subtype fun x => Membership.mem U x), Eq (((AlgebraicGeometry …
    i : Subtype fun x => Membership.mem U x
    ⊢ Eq (((AlgebraicGeometry.SheafedSpace.sheaf X).val.map (iVU i).op).hom (HMul. …
  -/
  rw [RingHom.map_one, RingHom.map_mul, gl_spec]
  /-
    case intro.intro.h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    h : ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), IsUnit ((X.preshe …
    V : (Subtype fun x => Membership.mem U x) → TopologicalSpace.Opens ↑↑X.toPresh …
    iVU : (x : Subtype fun x => Membership.mem U x) → Quiver.Hom (V x) U
    m : ∀ (x : Subtype fun x => Membership.mem U x), Membership.mem (V x) ↑x
    h_unit : ∀ (x : Subtype fun x => Membership.mem U x), IsUnit ((X.presheaf.map  …
    hcover : LE.le U (iSup V)
    g : (x : Subtype fun x => Membership.mem U x) → ↑(X.presheaf.obj { unop := V x …
    hg : ∀ (x : Subtype fun x => Membership.mem U x), Eq (HMul.hMul ((X.presheaf.m …
    ic : TopCat.Presheaf.IsCompatible (AlgebraicGeometry.SheafedSpace.sheaf X).val …
    gl : ↑(X.presheaf.obj { unop := U })
    gl_spec : ∀ (i : Subtype fun x => Membership.mem U x), Eq (((AlgebraicGeometry …
    i : Subtype fun x => Membership.mem U x
    ⊢ Eq (HMul.hMul (((AlgebraicGeometry.SheafedSpace.sheaf X).val.map (iVU i).op) …
  -/
  exact hg i
  /-
    🎉 no goals
  -/


/-- The basic open of a section `f` is the set of all points `x`, such that the germ of `f` at
`x` is a unit.
-/
def basicOpen {U : Opens X} (f : X.presheaf.obj (op U)) : Opens X where
  carrier := { x : X | ∃ (hx : x ∈ U), IsUnit (X.presheaf.germ U x hx f) }
  is_open' := by
    /-
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      ⊢ IsOpen (setOf fun x => Exists fun hx => IsUnit ((X.presheaf.germ U x hx).hom …
    -/
    rw [isOpen_iff_forall_mem_open]
    /-
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      ⊢ ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (setOf fun x => Exists fun hx  …
    -/
    rintro x ⟨hxU, hx⟩
    /-
      case intro
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
      ⊢ Exists fun t => And (HasSubset.Subset t (setOf fun x => Exists fun hx => IsU …
    -/
    obtain ⟨V, i, hxV, hf⟩ := X.isUnit_res_of_isUnit_germ U f x hxU hx
    /-
      case intro.intro.intro.intro
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      i : Quiver.Hom V U
      hxV : Membership.mem V x
      hf : IsUnit ((X.presheaf.map i.op).hom f)
      ⊢ Exists fun t => And (HasSubset.Subset t (setOf fun x => Exists fun hx => IsU …
    -/
    use V.1
    /-
      case h
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      i : Quiver.Hom V U
      hxV : Membership.mem V x
      hf : IsUnit ((X.presheaf.map i.op).hom f)
      ⊢ And (HasSubset.Subset V.carrier (setOf fun x => Exists fun hx => IsUnit ((X. …
    -/
    refine ⟨?_, V.2, hxV⟩
    /-
      case h
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      i : Quiver.Hom V U
      hxV : Membership.mem V x
      hf : IsUnit ((X.presheaf.map i.op).hom f)
      ⊢ HasSubset.Subset V.carrier (setOf fun x => Exists fun hx => IsUnit ((X.presh …
    -/
    intro y hy
    /-
      case h
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      i : Quiver.Hom V U
      hxV : Membership.mem V x
      hf : IsUnit ((X.presheaf.map i.op).hom f)
      y : ↑↑X.toPresheafedSpace
      hy : Membership.mem V.carrier y
      ⊢ Membership.mem (setOf fun x => Exists fun hx => IsUnit ((X.presheaf.germ U x …
    -/
    use i.le hy
    /-
      case h
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      i : Quiver.Hom V U
      hxV : Membership.mem V x
      hf : IsUnit ((X.presheaf.map i.op).hom f)
      y : ↑↑X.toPresheafedSpace
      hy : Membership.mem V.carrier y
      ⊢ IsUnit ((X.presheaf.germ U y ⋯).hom f)
    -/
    convert RingHom.isUnit_map (X.presheaf.germ _ y hy).hom hf
    /-
      case h.e'_3
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem U x
      hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      i : Quiver.Hom V U
      hxV : Membership.mem V x
      hf : IsUnit ((X.presheaf.map i.op).hom f)
      y : ↑↑X.toPresheafedSpace
      hy : Membership.mem V.carrier y
      ⊢ Eq ((X.presheaf.germ U y ⋯).hom f) ((X.presheaf.germ V y hy).hom ((X.preshea …
    -/
    exact (X.presheaf.germ_res_apply i y hy f).symm
    /-
      🎉 no goals
    -/


theorem mem_basicOpen {U : Opens X} (f : X.presheaf.obj (op U)) (x : X) (hx : x ∈ U) :
    x ∈ X.basicOpen f ↔ IsUnit (X.presheaf.germ U x hx f) :=
  ⟨Exists.choose_spec, (⟨hx, ·⟩)⟩


/-- A variant of `mem_basicOpen` with bundled `x : U`. -/
@[simp]
theorem mem_basicOpen' {U : Opens X} (f : X.presheaf.obj (op U)) (x : U) :
    ↑x ∈ X.basicOpen f ↔ IsUnit (X.presheaf.germ U x.1 x.2 f) :=
  mem_basicOpen X f x.1 x.2


@[simp]
theorem mem_top_basicOpen (f : X.presheaf.obj (op ⊤)) (x : X) :
    x ∈ X.basicOpen f ↔ IsUnit (X.presheaf.Γgerm x f) :=
  mem_basicOpen X f x .intro


theorem basicOpen_le {U : Opens X} (f : X.presheaf.obj (op U)) : X.basicOpen f ≤ U := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ LE.le (X.basicOpen f) U
  -/
  rintro x ⟨h, _⟩; exact h
                   /-
                     🎉 no goals
                   -/


/-- The restriction of a section `f` to the basic open of `f` is a unit. -/
theorem isUnit_res_basicOpen {U : Opens X} (f : X.presheaf.obj (op U)) :
    IsUnit (X.presheaf.map (@homOfLE (Opens X) _ _ _ (X.basicOpen_le f)).op f) := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ IsUnit ((X.presheaf.map (CategoryTheory.homOfLE ⋯).op).hom f)
  -/
  apply isUnit_of_isUnit_germ
  /-
    case h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem (X.basicOpen f) x), IsUni …
  -/
  rintro x ⟨hxU, hx⟩
  /-
    case h.intro
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
    ⊢ IsUnit ((X.presheaf.germ (X.basicOpen f) x ⋯).hom ((X.presheaf.map (Category …
  -/
  convert hx
  /-
    case h.e'_3
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    hx : IsUnit ((X.presheaf.germ U x hxU).hom f)
    ⊢ Eq ((X.presheaf.germ (X.basicOpen f) x ⋯).hom ((X.presheaf.map (CategoryTheo …
  -/
  exact X.presheaf.germ_res_apply _ _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem basicOpen_res {U V : (Opens X)ᵒᵖ} (i : U ⟶ V) (f : X.presheaf.obj U) :
    @basicOpen X (unop V) (X.presheaf.map i f) = unop V ⊓ @basicOpen X (unop U) f := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
    i : Quiver.Hom U V
    f : ↑(X.presheaf.obj U)
    ⊢ Eq (X.basicOpen ((X.presheaf.map i).hom f)) (Min.min (Opposite.unop V) (X.ba …
  -/
  ext x; constructor
    /-
      case h.h.mp
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      f : ↑(X.presheaf.obj U)
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (↑(X.basicOpen ((X.presheaf.map i).hom f))) x → Membership.me …
    -/
  · rintro ⟨hxV, hx⟩
    /-
      case h.h.mp.intro
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      f : ↑(X.presheaf.obj U)
      x : ↑↑X.toPresheafedSpace
      hxV : Membership.mem (Opposite.unop V) x
      hx : IsUnit ((X.presheaf.germ (Opposite.unop V) x hxV).hom ((X.presheaf.map i) …
      ⊢ Membership.mem (↑(Min.min (Opposite.unop V) (X.basicOpen f))) x
    -/
    rw [CommRingCat.germ_res_apply' X.presheaf] at hx
    /-
      case h.h.mp.intro
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      f : ↑(X.presheaf.obj U)
      x : ↑↑X.toPresheafedSpace
      hxV : Membership.mem (Opposite.unop V) x
      hx : IsUnit ((X.presheaf.germ (Opposite.unop U) x ⋯).hom f)
      ⊢ Membership.mem (↑(Min.min (Opposite.unop V) (X.basicOpen f))) x
    -/
    exact ⟨hxV, i.unop.le hxV, hx⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      f : ↑(X.presheaf.obj U)
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (↑(Min.min (Opposite.unop V) (X.basicOpen f))) x → Membership …
    -/
  · rintro ⟨hxV, _, hx⟩
    /-
      case h.h.mpr.intro.intro
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      f : ↑(X.presheaf.obj U)
      x : ↑↑X.toPresheafedSpace
      hxV : Membership.mem (↑(Opposite.unop V)) x
      w✝ : Membership.mem (Opposite.unop U) x
      hx : IsUnit ((X.presheaf.germ (Opposite.unop U) x w✝).hom f)
      ⊢ Membership.mem (↑(X.basicOpen ((X.presheaf.map i).hom f))) x
    -/
    refine ⟨hxV, ?_⟩
    /-
      case h.h.mpr.intro.intro
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      f : ↑(X.presheaf.obj U)
      x : ↑↑X.toPresheafedSpace
      hxV : Membership.mem (↑(Opposite.unop V)) x
      w✝ : Membership.mem (Opposite.unop U) x
      hx : IsUnit ((X.presheaf.germ (Opposite.unop U) x w✝).hom f)
      ⊢ IsUnit ((X.presheaf.germ (Opposite.unop V) x hxV).hom ((X.presheaf.map i).ho …
    -/
    rw [CommRingCat.germ_res_apply' X.presheaf]
    /-
      case h.h.mpr.intro.intro
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      f : ↑(X.presheaf.obj U)
      x : ↑↑X.toPresheafedSpace
      hxV : Membership.mem (↑(Opposite.unop V)) x
      w✝ : Membership.mem (Opposite.unop U) x
      hx : IsUnit ((X.presheaf.germ (Opposite.unop U) x w✝).hom f)
      ⊢ IsUnit ((X.presheaf.germ (Opposite.unop U) x ⋯).hom f)
    -/
    exact hx
    /-
      🎉 no goals
    -/

-- This should fire before `basicOpen_res`.
-- Porting note: this lemma is not in simple normal form because of `basicOpen_res`, as in Lean3
-- it is specifically said "This should fire before `basic_open_res`", this lemma is marked with
-- high priority

@[simp (high)]
theorem basicOpen_res_eq {U V : (Opens X)ᵒᵖ} (i : U ⟶ V) [IsIso i] (f : X.presheaf.obj U) :
    @basicOpen X (unop V) (X.presheaf.map i f) = @RingedSpace.basicOpen X (unop U) f := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
    i : Quiver.Hom U V
    inst✝ : CategoryTheory.IsIso i
    f : ↑(X.presheaf.obj U)
    ⊢ Eq (X.basicOpen ((X.presheaf.map i).hom f)) (X.basicOpen f)
  -/
  apply le_antisymm
    /-
      case a
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      inst✝ : CategoryTheory.IsIso i
      f : ↑(X.presheaf.obj U)
      ⊢ LE.le (X.basicOpen ((X.presheaf.map i).hom f)) (X.basicOpen f)
    -/
  · rw [X.basicOpen_res i f]; exact inf_le_right
                              /-
                                🎉 no goals
                              -/
    /-
      case a
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      inst✝ : CategoryTheory.IsIso i
      f : ↑(X.presheaf.obj U)
      ⊢ LE.le (X.basicOpen f) (X.basicOpen ((X.presheaf.map i).hom f))
    -/
  · have := X.basicOpen_res (inv i) (X.presheaf.map i f)
    rw [← CommRingCat.comp_apply, ← X.presheaf.map_comp, IsIso.hom_inv_id, X.presheaf.map_id,
        CommRingCat.id_apply] at this
    /-
      case a
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      inst✝ : CategoryTheory.IsIso i
      f : ↑(X.presheaf.obj U)
      this : Eq (X.basicOpen f) (Min.min (Opposite.unop U) (X.basicOpen ((X.presheaf …
      ⊢ LE.le (X.basicOpen f) (X.basicOpen ((X.presheaf.map i).hom f))
    -/
    rw [this]
    /-
      case a
      X : AlgebraicGeometry.RingedSpace
      U V : Opposite (TopologicalSpace.Opens ↑↑X.toPresheafedSpace)
      i : Quiver.Hom U V
      inst✝ : CategoryTheory.IsIso i
      f : ↑(X.presheaf.obj U)
      this : Eq (X.basicOpen f) (Min.min (Opposite.unop U) (X.basicOpen ((X.presheaf …
      ⊢ LE.le (Min.min (Opposite.unop U) (X.basicOpen ((X.presheaf.map i).hom f))) ( …
    -/
    exact inf_le_right
    /-
      🎉 no goals
    -/


@[simp]
theorem basicOpen_mul {U : Opens X} (f g : X.presheaf.obj (op U)) :
    X.basicOpen (f * g) = X.basicOpen f ⊓ X.basicOpen g := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f g : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq (X.basicOpen (HMul.hMul f g)) (Min.min (X.basicOpen f) (X.basicOpen g))
  -/
  ext x
  /-
    case h.h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f g : ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    ⊢ Iff (Membership.mem (↑(X.basicOpen (HMul.hMul f g))) x) (Membership.mem (↑(M …
  -/
  by_cases hx : x ∈ U
    /-
      case pos
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f g : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U x
      ⊢ Iff (Membership.mem (↑(X.basicOpen (HMul.hMul f g))) x) (Membership.mem (↑(M …
    -/
  · simp [mem_basicOpen (hx := hx)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f g : ↑(X.presheaf.obj { unop := U })
      x : ↑↑X.toPresheafedSpace
      hx : Not (Membership.mem U x)
      ⊢ Iff (Membership.mem (↑(X.basicOpen (HMul.hMul f g))) x) (Membership.mem (↑(M …
    -/
  · simp [mt (basicOpen_le X _ ·) hx]
    /-
      🎉 no goals
    -/


@[simp]
lemma basicOpen_pow {U : Opens X} (f : X.presheaf.obj (op U)) (n : ℕ) (h : 0 < n) :
    X.basicOpen (f ^ n) = X.basicOpen f := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    n : Nat
    h : LT.lt 0 n
    ⊢ Eq (X.basicOpen (HPow.hPow f n)) (X.basicOpen f)
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le' h
  induction k with
  | zero => simp
  | succ n hn => rw [pow_add]; simp_all


theorem basicOpen_of_isUnit {U : Opens X} {f : X.presheaf.obj (op U)} (hf : IsUnit f) :
    X.basicOpen f = U := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    hf : IsUnit f
    ⊢ Eq (X.basicOpen f) U
  -/
  apply le_antisymm
    /-
      case a
      X : AlgebraicGeometry.RingedSpace
      U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      f : ↑(X.presheaf.obj { unop := U })
      hf : IsUnit f
      ⊢ LE.le (X.basicOpen f) U
    -/
  · exact X.basicOpen_le f
    /-
      🎉 no goals
    -/
  /-
    case a
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    hf : IsUnit f
    ⊢ LE.le U (X.basicOpen f)
  -/
  intro x hx
  /-
    case a
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    hf : IsUnit f
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (↑U) x
    ⊢ Membership.mem (↑(X.basicOpen f)) x
  -/
  rw [SetLike.mem_coe, X.mem_basicOpen f x hx]
  /-
    case a
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    hf : IsUnit f
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem (↑U) x
    ⊢ IsUnit ((X.presheaf.germ U x hx).hom f)
  -/
  exact RingHom.isUnit_map _ hf
  /-
    🎉 no goals
  -/


/--
The zero locus of a set of sections `s` over an open set `U` is the closed set consisting of
the complement of `U` and of all points of `U`, where all elements of `f` vanish.
-/
def zeroLocus {U : Opens X} (s : Set (X.presheaf.obj (op U))) : Set X :=
  ⋂ f ∈ s, (X.basicOpen f)ᶜ


lemma zeroLocus_isClosed {U : Opens X} (s : Set (X.presheaf.obj (op U))) :
    IsClosed (X.zeroLocus s) := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    s : Set ↑(X.presheaf.obj { unop := U })
    ⊢ IsClosed (X.zeroLocus s)
  -/
  apply isClosed_biInter
  /-
    case h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    s : Set ↑(X.presheaf.obj { unop := U })
    ⊢ ∀ (i : ↑(X.presheaf.obj { unop := U })), Membership.mem s i → IsClosed (HasC …
  -/
  intro i _
  /-
    case h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    s : Set ↑(X.presheaf.obj { unop := U })
    i : ↑(X.presheaf.obj { unop := U })
    a✝ : Membership.mem s i
    ⊢ IsClosed (HasCompl.compl ↑(X.basicOpen i))
  -/
  simp only [isClosed_compl_iff]
  /-
    case h
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    s : Set ↑(X.presheaf.obj { unop := U })
    i : ↑(X.presheaf.obj { unop := U })
    a✝ : Membership.mem s i
    ⊢ IsOpen ↑(X.basicOpen i)
  -/
  exact Opens.isOpen (X.basicOpen i)
  /-
    🎉 no goals
  -/


lemma zeroLocus_singleton {U : Opens X} (f : X.presheaf.obj (op U)) :
    X.zeroLocus {f} = (X.basicOpen f).carrierᶜ := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    f : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq (X.zeroLocus (Singleton.singleton f)) (HasCompl.compl (X.basicOpen f).car …
  -/
  simp [zeroLocus]
  /-
    🎉 no goals
  -/


@[simp]
lemma zeroLocus_empty_eq_univ {U : Opens X} :
    X.zeroLocus (∅ : Set (X.presheaf.obj (op U))) = Set.univ := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    ⊢ Eq (X.zeroLocus EmptyCollection.emptyCollection) Set.univ
  -/
  simp [zeroLocus]
  /-
    🎉 no goals
  -/


@[simp]
lemma mem_zeroLocus_iff {U : Opens X} (s : Set (X.presheaf.obj (op U))) (x : X) :
    x ∈ X.zeroLocus s ↔ ∀ f ∈ s, x ∉ X.basicOpen f := by
  /-
    X : AlgebraicGeometry.RingedSpace
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    s : Set ↑(X.presheaf.obj { unop := U })
    x : ↑↑X.toPresheafedSpace
    ⊢ Iff (Membership.mem (X.zeroLocus s) x) (∀ (f : ↑(X.presheaf.obj { unop := U  …
  -/
  simp [zeroLocus]
  /-
    🎉 no goals
  -/


