/-- The colimit of `F ⋙ forget₂ (ModuleCat R) AddCommGrp` in the category `AddCommGrp`.
In the following, we will show that this has the structure of an `R`-module.
-/
abbrev M : AddCommGrp :=
  AddCommGrp.FilteredColimits.colimit.{v, u}
    (F ⋙ forget₂ (ModuleCat R) AddCommGrp.{max v u})


/-- The canonical projection into the colimit, as a quotient type. -/
abbrev M.mk : (Σ j, F.obj j) → M F :=
  Quot.mk (Types.Quot.Rel (F ⋙ forget (ModuleCat R)))


theorem M.mk_eq (x y : Σ j, F.obj j)
    (h : ∃ (k : J) (f : x.1 ⟶ k) (g : y.1 ⟶ k), F.map f x.2 = F.map g y.2) : M.mk F x = M.mk F y :=
  Quot.eqvGen_sound (Types.FilteredColimit.eqvGen_quot_rel_of_rel (F ⋙ forget (ModuleCat R)) x y h)


/-- The "unlifted" version of scalar multiplication in the colimit. -/
def colimitSMulAux (r : R) (x : Σ j, F.obj j) : M F :=
  M.mk F ⟨x.1, r • x.2⟩


theorem colimitSMulAux_eq_of_rel (r : R) (x y : Σ j, F.obj j)
    (h : Types.FilteredColimit.Rel (F ⋙ forget (ModuleCat R)) x y) :
    colimitSMulAux F r x = colimitSMulAux F r y := by
  /-
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    r : R
    x y : Sigma fun j => ↑(F.obj j)
    h : CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory.fo …
    ⊢ Eq (ModuleCat.FilteredColimits.colimitSMulAux F r x) (ModuleCat.FilteredColi …
  -/
  apply M.mk_eq
  /-
    case h
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    r : R
    x y : Sigma fun j => ↑(F.obj j)
    h : CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory.fo …
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f).hom ⟨x.fst, HS …
  -/
  obtain ⟨k, f, g, hfg⟩ := h
  /-
    case h.intro.intro.intro
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    r : R
    x y : Sigma fun j => ↑(F.obj j)
    k : J
    f : Quiver.Hom x.fst k
    g : Quiver.Hom y.fst k
    hfg : Eq ((F.comp (CategoryTheory.forget (ModuleCat R))).map f x.snd) ((F.comp …
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f).hom ⟨x.fst, HS …
  -/
  use k, f, g
  /-
    case h
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    r : R
    x y : Sigma fun j => ↑(F.obj j)
    k : J
    f : Quiver.Hom x.fst k
    g : Quiver.Hom y.fst k
    hfg : Eq ((F.comp (CategoryTheory.forget (ModuleCat R))).map f x.snd) ((F.comp …
    ⊢ Eq ((F.map f).hom ⟨x.fst, HSMul.hSMul r x.snd⟩.snd) ((F.map g).hom ⟨y.fst, H …
  -/
  simp only [Functor.comp_obj, Functor.comp_map, forget_map] at hfg
  /-
    case h
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    r : R
    x y : Sigma fun j => ↑(F.obj j)
    k : J
    f : Quiver.Hom x.fst k
    g : Quiver.Hom y.fst k
    hfg : Eq ((F.map f).hom x.snd) ((F.map g).hom y.snd)
    ⊢ Eq ((F.map f).hom ⟨x.fst, HSMul.hSMul r x.snd⟩.snd) ((F.map g).hom ⟨y.fst, H …
  -/
  simp [hfg]
  /-
    🎉 no goals
  -/


/-- Scalar multiplication in the colimit. See also `colimitSMulAux`. -/
instance colimitHasSMul : SMul R (M F) where
  smul r x := by
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      x : ↑(ModuleCat.FilteredColimits.M F)
      ⊢ ↑(ModuleCat.FilteredColimits.M F)
    -/
    refine Quot.lift (colimitSMulAux F r) ?_ x
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      x : ↑(ModuleCat.FilteredColimits.M F)
      ⊢ ∀ (a b : Sigma fun j => ↑(F.obj j)), CategoryTheory.Limits.Types.Quot.Rel (( …
    -/
    intro x y h
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      x✝ : ↑(ModuleCat.FilteredColimits.M F)
      x y : Sigma fun j => ↑(F.obj j)
      h : CategoryTheory.Limits.Types.Quot.Rel ((((F.comp (CategoryTheory.forget₂ (M …
      ⊢ Eq (ModuleCat.FilteredColimits.colimitSMulAux F r x) (ModuleCat.FilteredColi …
    -/
    apply colimitSMulAux_eq_of_rel
    /-
      case h
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      x✝ : ↑(ModuleCat.FilteredColimits.M F)
      x y : Sigma fun j => ↑(F.obj j)
      h : CategoryTheory.Limits.Types.Quot.Rel ((((F.comp (CategoryTheory.forget₂ (M …
      ⊢ CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory.forg …
    -/
    apply Types.FilteredColimit.rel_of_quot_rel
    /-
      case h.a
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      x✝ : ↑(ModuleCat.FilteredColimits.M F)
      x y : Sigma fun j => ↑(F.obj j)
      h : CategoryTheory.Limits.Types.Quot.Rel ((((F.comp (CategoryTheory.forget₂ (M …
      ⊢ CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget (ModuleC …
    -/
    exact h
    /-
      🎉 no goals
    -/


@[simp]
theorem colimit_smul_mk_eq (r : R) (x : Σ j, F.obj j) : r • M.mk F x = M.mk F ⟨x.1, r • x.2⟩ :=
  rfl


private theorem colimitModule.one_smul (x : (M F)) : (1 : R) • x = x := by
  /-
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    x : ↑(ModuleCat.FilteredColimits.M F)
    ⊢ Eq (HSMul.hSMul 1 x) x
  -/
  refine Quot.inductionOn x ?_; clear x; intro x; obtain ⟨j, x⟩ := x
  /-
    case mk
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    j : J
    x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
    ⊢ Eq (HSMul.hSMul 1 (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((((F.comp  …
  -/
  erw [colimit_smul_mk_eq F 1 ⟨j, x⟩]
  /-
    case mk
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    j : J
    x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
    ⊢ Eq (ModuleCat.FilteredColimits.M.mk F ⟨⟨j, x⟩.fst, HSMul.hSMul 1 ⟨j, x⟩.snd⟩ …
  -/
  simp
  /-
    case mk
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    j : J
    x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
    ⊢ Eq (ModuleCat.FilteredColimits.M.mk F ⟨j, x⟩) (Quot.mk (CategoryTheory.Limit …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/pull/11083): writing directly the `Module` instance makes things very slow.

instance colimitMulAction : MulAction R (M F) where
  one_smul x := by
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      x : ↑(ModuleCat.FilteredColimits.M F)
      ⊢ Eq (HSMul.hSMul 1 x) x
    -/
    refine Quot.inductionOn x ?_; clear x; intro x; obtain ⟨j, x⟩ := x
    /-
      case mk
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      j : J
      x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
      ⊢ Eq (HSMul.hSMul 1 (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((((F.comp  …
    -/
    erw [colimit_smul_mk_eq F 1 ⟨j, x⟩, one_smul]
    /-
      case mk
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      j : J
      x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
      ⊢ Eq (ModuleCat.FilteredColimits.M.mk F ⟨⟨j, x⟩.fst, ⟨j, x⟩.snd⟩) (Quot.mk (Ca …
    -/
    rfl
    /-
      🎉 no goals
    -/
  mul_smul r s x := by
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r s : R
      x : ↑(ModuleCat.FilteredColimits.M F)
      ⊢ Eq (HSMul.hSMul (HMul.hMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
    -/
    refine Quot.inductionOn x ?_; clear x; intro x; obtain ⟨j, x⟩ := x
    erw [colimit_smul_mk_eq F (r * s) ⟨j, x⟩, colimit_smul_mk_eq F s ⟨j, x⟩,
      colimit_smul_mk_eq F r ⟨j, _⟩, mul_smul]


instance colimitSMulWithZero : SMulWithZero R (M F) :=
{ colimitMulAction F with
  smul_zero := fun r => by
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      ⊢ Eq (HSMul.hSMul r 0) 0
    -/
    erw [colimit_zero_eq _ (IsFiltered.nonempty.some : J), colimit_smul_mk_eq, smul_zero]
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      ⊢ Eq (ModuleCat.FilteredColimits.M.mk F ⟨⟨⋯.some, 0⟩.fst, 0⟩) (AddMonCat.Filte …
    -/
    rfl
    /-
      🎉 no goals
    -/
  zero_smul := fun x => by
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      x : ↑(ModuleCat.FilteredColimits.M F)
      ⊢ Eq (HSMul.hSMul 0 x) 0
    -/
    refine Quot.inductionOn x ?_; clear x; intro x; obtain ⟨j, x⟩ := x
    /-
      case mk
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      j : J
      x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
      ⊢ Eq (HSMul.hSMul 0 (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((((F.comp  …
    -/
    erw [colimit_smul_mk_eq, zero_smul, colimit_zero_eq _ j]
    /-
      case mk
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      j : J
      x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
      ⊢ Eq (ModuleCat.FilteredColimits.M.mk F ⟨⟨j, x⟩.fst, 0⟩) (AddMonCat.FilteredCo …
    -/
    rfl }
    /-
      🎉 no goals
    -/


private theorem colimitModule.add_smul (r s : R) (x : (M F)) : (r + s) • x = r • x + s • x := by
  /-
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    r s : R
    x : ↑(ModuleCat.FilteredColimits.M F)
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
  -/
  refine Quot.inductionOn x ?_; clear x; intro x; obtain ⟨j, x⟩ := x
  erw [colimit_smul_mk_eq, _root_.add_smul, colimit_smul_mk_eq, colimit_smul_mk_eq,
      colimit_add_mk_eq _ ⟨j, _⟩ ⟨j, _⟩ j (𝟙 j) (𝟙 j)]
  simp only [Functor.comp_obj, forget₂_obj, Functor.comp_map, CategoryTheory.Functor.map_id,
    forget₂_map]
  /-
    case mk
    R : Type u
    inst✝² : Ring R
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J (ModuleCatMax R)
    r s : R
    j : J
    x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
    ⊢ Eq (ModuleCat.FilteredColimits.M.mk F ⟨j, HAdd.hAdd (HSMul.hSMul r x) (HSMul …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance colimitModule : Module R (M F) :=
{ colimitMulAction F,
  colimitSMulWithZero F with
  smul_add := fun r x y => by
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      x y : ↑(ModuleCat.FilteredColimits.M F)
      ⊢ Eq (HSMul.hSMul r (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
    -/
    refine Quot.induction_on₂ x y ?_; clear x y; intro x y; obtain ⟨i, x⟩ := x; obtain ⟨j, y⟩ := y
    erw [colimit_add_mk_eq _ ⟨i, _⟩ ⟨j, _⟩ (max' i j) (IsFiltered.leftToMax i j)
      (IsFiltered.rightToMax i j), colimit_smul_mk_eq, smul_add, colimit_smul_mk_eq,
      colimit_smul_mk_eq, colimit_add_mk_eq _ ⟨i, _⟩ ⟨j, _⟩ (max' i j) (IsFiltered.leftToMax i j)
      (IsFiltered.rightToMax i j), LinearMap.map_smul, LinearMap.map_smul]
    /-
      case mk.mk
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J (ModuleCatMax R)
      r : R
      i : J
      x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
      j : J
      y : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
      ⊢ Eq (ModuleCat.FilteredColimits.M.mk F ⟨⟨CategoryTheory.IsFiltered.max i j, H …
    -/
    rfl
    /-
      🎉 no goals
    -/
  add_smul := colimitModule.add_smul F }


/-- The bundled `R`-module giving the filtered colimit of a diagram. -/
def colimit : ModuleCatMax.{v, u, u} R :=
  ModuleCat.of R (M F)


/-- The linear map from a given `R`-module in the diagram to the colimit module. -/
def coconeMorphism (j : J) : F.obj j ⟶ colimit F :=
  ofHom
    { (AddCommGrp.FilteredColimits.colimitCocone
      (F ⋙ forget₂ (ModuleCat R) AddCommGrp.{max v u})).ι.app j with
                               /-
                                 R : Type u
                                 inst✝² : Ring R
                                 J : Type v
                                 inst✝¹ : CategoryTheory.SmallCategory J
                                 inst✝ : CategoryTheory.IsFiltered J
                                 F : CategoryTheory.Functor J (ModuleCatMax R)
                                 j : J
                                 r : R
                                 x : ↑(F.toPrefunctor.1 j)
                                 ⊢ Eq ({ toFun := (↑__src✝).toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) (HS …
                               -/
    map_smul' := fun r x => by erw [colimit_smul_mk_eq F r ⟨j, x⟩]; rfl }
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The cocone over the proposed colimit module. -/
def colimitCocone : Cocone F where
  pt := colimit F
  ι :=
    { app := coconeMorphism F
      naturality := fun _ _' f =>
        hom_ext <| LinearMap.coe_injective
          ((Types.TypeMax.colimitCocone (F ⋙ forget (ModuleCat R))).ι.naturality f) }


/-- Given a cocone `t` of `F`, the induced monoid linear map from the colimit to the cocone point.
We already know that this is a morphism between additive groups. The only thing left to see is that
it is a linear map, i.e. preserves scalar multiplication.
-/
def colimitDesc (t : Cocone F) : colimit F ⟶ t.pt :=
  ofHom
    { (AddCommGrp.FilteredColimits.colimitCoconeIsColimit
          (F ⋙ forget₂ (ModuleCatMax.{v, u} R) AddCommGrp.{max v u})).desc
      ((forget₂ (ModuleCat R) AddCommGrp.{max v u}).mapCocone t) with
    map_smul' := fun r x => by
      /-
        R : Type u
        inst✝² : Ring R
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J (ModuleCatMax R)
        t : CategoryTheory.Limits.Cocone F
        r : R
        x : ↑(ModuleCat.FilteredColimits.M F)
        ⊢ Eq ({ toFun := (↑__src✝).toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) (HS …
      -/
      refine Quot.inductionOn x ?_; clear x; intro x; obtain ⟨j, x⟩ := x
      /-
        case mk
        R : Type u
        inst✝² : Ring R
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J (ModuleCatMax R)
        t : CategoryTheory.Limits.Cocone F
        r : R
        j : J
        x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
        ⊢ Eq ({ toFun := (↑__src✝).toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r (Quot.m …
      -/
      erw [colimit_smul_mk_eq]
      /-
        case mk
        R : Type u
        inst✝² : Ring R
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J (ModuleCatMax R)
        t : CategoryTheory.Limits.Cocone F
        r : R
        j : J
        x : ((((F.comp (CategoryTheory.forget₂ (ModuleCat R) AddCommGrp)).comp (Catego …
        ⊢ Eq ({ toFun := (↑__src✝).toFun, map_add' := ⋯ }.toFun (ModuleCat.FilteredCol …
      -/
      exact LinearMap.map_smul (t.ι.app j).hom r x }
      /-
        🎉 no goals
      -/


/-- The proposed colimit cocone is a colimit in `ModuleCat R`. -/
def colimitCoconeIsColimit : IsColimit (colimitCocone F) where
  desc := colimitDesc F
  fac t j :=
    hom_ext <| LinearMap.coe_injective <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget (ModuleCat R))).fac
        ((forget (ModuleCat R)).mapCocone t) j
  uniq t _ h :=
    hom_ext <| LinearMap.coe_injective <|
      (Types.TypeMax.colimitCoconeIsColimit (F ⋙ forget (ModuleCat R))).uniq
        ((forget (ModuleCat R)).mapCocone t) _ fun j => funext fun x => LinearMap.congr_fun
          (ModuleCat.hom_ext_iff.mp (h j)) x


instance forget₂AddCommGroup_preservesFilteredColimits :
    PreservesFilteredColimits (forget₂ (ModuleCat.{u} R) AddCommGrp.{u}) where
  preserves_filtered_colimits J _ _ :=
  { -- Porting note: without the curly braces for `F`
    -- here we get a confusing error message about universes.
    preservesColimit := fun {F : J ⥤ ModuleCat.{u} R} =>
      preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit F)
        (AddCommGrp.FilteredColimits.colimitCoconeIsColimit
          (F ⋙ forget₂ (ModuleCat.{u} R) AddCommGrp.{u})) }


instance forget_preservesFilteredColimits : PreservesFilteredColimits (forget (ModuleCat.{u} R)) :=
  Limits.comp_preservesFilteredColimits (forget₂ (ModuleCat R) AddCommGrp)
    (forget AddCommGrp)


instance forget_reflectsFilteredColimits : ReflectsFilteredColimits (forget (ModuleCat.{u} R)) where
  reflects_filtered_colimits _ := { reflectsColimit := reflectsColimit_of_reflectsIsomorphisms _ _ }


