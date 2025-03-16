/-- The colimit of `F ⋙ forget MonCat` in the category of types.
In the following, we will construct a monoid structure on `M`.
-/
@[to_additive
      "The colimit of `F ⋙ forget AddMon` in the category of types.
      In the following, we will construct an additive monoid structure on `M`."]
abbrev M :=
  Types.Quot (F ⋙ forget MonCat)


/-- The canonical projection into the colimit, as a quotient type. -/
@[to_additive "The canonical projection into the colimit, as a quotient type."]
noncomputable abbrev M.mk : (Σ j, F.obj j) → M.{v, u} F :=
  Quot.mk _


@[to_additive]
theorem M.mk_eq (x y : Σ j, F.obj j)
    (h : ∃ (k : J) (f : x.1 ⟶ k) (g : y.1 ⟶ k), F.map f x.2 = F.map g y.2) :
    M.mk.{v, u} F x = M.mk F y :=
  Quot.eqvGen_sound (Types.FilteredColimit.eqvGen_quot_rel_of_rel (F ⋙ forget MonCat) x y h)


/-- As `J` is nonempty, we can pick an arbitrary object `j₀ : J`. We use this object to define the
"one" in the colimit as the equivalence class of `⟨j₀, 1 : F.obj j₀⟩`.
-/
@[to_additive
  "As `J` is nonempty, we can pick an arbitrary object `j₀ : J`. We use this object to
  define the \"zero\" in the colimit as the equivalence class of `⟨j₀, 0 : F.obj j₀⟩`."]
noncomputable instance colimitOne :
  One (M.{v, u} F) where one := M.mk F ⟨IsFiltered.nonempty.some,1⟩


/-- The definition of the "one" in the colimit is independent of the chosen object of `J`.
In particular, this lemma allows us to "unfold" the definition of `colimit_one` at a custom chosen
object `j`.
-/
@[to_additive
      "The definition of the \"zero\" in the colimit is independent of the chosen object
      of `J`. In particular, this lemma allows us to \"unfold\" the definition of `colimit_zero` at
      a custom chosen object `j`."]
theorem colimit_one_eq (j : J) : (1 : M.{v, u} F) = M.mk F ⟨j, 1⟩ := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j : J
    ⊢ Eq 1 (MonCat.FilteredColimits.M.mk F ⟨j, 1⟩)
  -/
  apply M.mk_eq
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j : J
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) ⟨⋯.some, 1⟩.sn …
  -/
  refine ⟨max' _ j, IsFiltered.leftToMax _ j, IsFiltered.rightToMax _ j, ?_⟩
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j : J
    ⊢ Eq ((F.map (CategoryTheory.IsFiltered.leftToMax ⟨⋯.some, 1⟩.fst j)) ⟨⋯.some, …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The "unlifted" version of multiplication in the colimit. To multiply two dependent pairs
`⟨j₁, x⟩` and `⟨j₂, y⟩`, we pass to a common successor of `j₁` and `j₂` (given by `IsFiltered.max`)
and multiply them there.
-/
@[to_additive
      "The \"unlifted\" version of addition in the colimit. To add two dependent pairs
      `⟨j₁, x⟩` and `⟨j₂, y⟩`, we pass to a common successor of `j₁` and `j₂`
      (given by `IsFiltered.max`) and add them there."]
noncomputable def colimitMulAux (x y : Σ j, F.obj j) : M.{v, u} F :=
  M.mk F ⟨IsFiltered.max x.fst y.fst, F.map (IsFiltered.leftToMax x.1 y.1) x.2 *
    F.map (IsFiltered.rightToMax x.1 y.1) y.2⟩


/-- Multiplication in the colimit is well-defined in the left argument. -/
@[to_additive "Addition in the colimit is well-defined in the left argument."]
theorem colimitMulAux_eq_of_rel_left {x x' y : Σ j, F.obj j}
    (hxx' : Types.FilteredColimit.Rel (F ⋙ forget MonCat) x x') :
    colimitMulAux.{v, u} F x y = colimitMulAux.{v, u} F x' y := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    x x' y : Sigma fun j => ↑(F.obj j)
    hxx' : CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory …
    ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F x y) (MonCat.FilteredColimits.co …
  -/
  obtain ⟨j₁, x⟩ := x; obtain ⟨j₂, y⟩ := y; obtain ⟨j₃, x'⟩ := x'
  /-
    case mk.mk.mk
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    hxx' : CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory …
    ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F ⟨j₁, x⟩ ⟨j₂, y⟩) (MonCat.Filtere …
  -/
  obtain ⟨l, f, g, hfg⟩ := hxx'
  /-
    case mk.mk.mk.intro.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.comp (CategoryTheory.forget MonCat)).map f ⟨j₁, x⟩.snd) ((F.comp  …
    ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F ⟨j₁, x⟩ ⟨j₂, y⟩) (MonCat.Filtere …
  -/
  simp? at hfg says simp only [Functor.comp_obj, Functor.comp_map, forget_map] at hfg
  obtain ⟨s, α, β, γ, h₁, h₂, h₃⟩ :=
    IsFiltered.tulip (IsFiltered.leftToMax j₁ j₂) (IsFiltered.rightToMax j₁ j₂)
      (IsFiltered.rightToMax j₃ j₂) (IsFiltered.leftToMax j₃ j₂) f g
  /-
    case mk.mk.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.map f) x) ((F.map g) x')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₃ j₂) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F ⟨j₁, x⟩ ⟨j₂, y⟩) (MonCat.Filtere …
  -/
  apply M.mk_eq
  /-
    case mk.mk.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.map f) x) ((F.map g) x')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₃ j₂) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) ⟨CategoryTheor …
  -/
  use s, α, γ
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.map f) x) ((F.map g) x')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₃ j₂) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    ⊢ Eq ((F.map α) ⟨CategoryTheory.IsFiltered.max ⟨j₁, x⟩.fst ⟨j₂, y⟩.fst, HMul.h …
  -/
  dsimp
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.map f) x) ((F.map g) x')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₃ j₂) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    ⊢ Eq ((F.map α) (HMul.hMul ((F.map (CategoryTheory.IsFiltered.leftToMax j₁ j₂) …
  -/
  simp_rw [MonoidHom.map_mul]
  -- Porting note: Lean cannot seem to use lemmas from concrete categories directly
  change (F.map _ ≫ F.map _) _ * (F.map _ ≫ F.map _) _ =
    (F.map _ ≫ F.map _) _ * (F.map _ ≫ F.map _) _
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.map f) x) ((F.map g) x')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₃ j₂) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    ⊢ Eq (HMul.hMul ((CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Is …
  -/
  simp_rw [← F.map_comp, h₁, h₂, h₃, F.map_comp]
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.map f) x) ((F.map g) x')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₃ j₂) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    ⊢ Eq (HMul.hMul ((CategoryTheory.CategoryStruct.comp (F.map f) (F.map β)) x) ( …
  -/
  congr 1
  /-
    case h.e_a
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.map f) x) ((F.map g) x')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₃ j₂) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F.map f) (F.map β)) x) ((CategoryTh …
  -/
  change F.map _ (F.map _ _) = F.map _ (F.map _ _)
  /-
    case h.e_a
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    x : ↑(F.obj j₁)
    j₂ : J
    y : ↑(F.obj j₂)
    j₃ : J
    x' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, x⟩.fst l
    g : Quiver.Hom ⟨j₃, x'⟩.fst l
    hfg : Eq ((F.map f) x) ((F.map g) x')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₃ j₂) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    ⊢ Eq ((F.map β) ((F.map f) x)) ((F.map β) ((F.map g) x'))
  -/
  rw [hfg]
  /-
    🎉 no goals
  -/


/-- Multiplication in the colimit is well-defined in the right argument. -/
@[to_additive "Addition in the colimit is well-defined in the right argument."]
theorem colimitMulAux_eq_of_rel_right {x y y' : Σ j, F.obj j}
    (hyy' : Types.FilteredColimit.Rel (F ⋙ forget MonCat) y y') :
    colimitMulAux.{v, u} F x y = colimitMulAux.{v, u} F x y' := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    x y y' : Sigma fun j => ↑(F.obj j)
    hyy' : CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory …
    ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F x y) (MonCat.FilteredColimits.co …
  -/
  obtain ⟨j₁, y⟩ := y; obtain ⟨j₂, x⟩ := x; obtain ⟨j₃, y'⟩ := y'
  /-
    case mk.mk.mk
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    hyy' : CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory …
    ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F ⟨j₂, x⟩ ⟨j₁, y⟩) (MonCat.Filtere …
  -/
  obtain ⟨l, f, g, hfg⟩ := hyy'
  /-
    case mk.mk.mk.intro.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.comp (CategoryTheory.forget MonCat)).map f ⟨j₁, y⟩.snd) ((F.comp  …
    ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F ⟨j₂, x⟩ ⟨j₁, y⟩) (MonCat.Filtere …
  -/
  simp only [Functor.comp_obj, Functor.comp_map, forget_map] at hfg
  obtain ⟨s, α, β, γ, h₁, h₂, h₃⟩ :=
    IsFiltered.tulip (IsFiltered.rightToMax j₂ j₁) (IsFiltered.leftToMax j₂ j₁)
      (IsFiltered.leftToMax j₂ j₃) (IsFiltered.rightToMax j₂ j₃) f g
  /-
    case mk.mk.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.map f) y) ((F.map g) y')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₁) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₃) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F ⟨j₂, x⟩ ⟨j₁, y⟩) (MonCat.Filtere …
  -/
  apply M.mk_eq
  /-
    case mk.mk.mk.intro.intro.intro.intro.intro.intro.intro.intro.intro.h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.map f) y) ((F.map g) y')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₁) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₃) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) ⟨CategoryTheor …
  -/
  use s, α, γ
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.map f) y) ((F.map g) y')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₁) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₃) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq ((F.map α) ⟨CategoryTheory.IsFiltered.max ⟨j₂, x⟩.fst ⟨j₁, y⟩.fst, HMul.h …
  -/
  dsimp
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.map f) y) ((F.map g) y')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₁) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₃) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq ((F.map α) (HMul.hMul ((F.map (CategoryTheory.IsFiltered.leftToMax j₂ j₁) …
  -/
  simp_rw [MonoidHom.map_mul]
  -- Porting note: Lean cannot seem to use lemmas from concrete categories directly
  change (F.map _ ≫ F.map _) _ * (F.map _ ≫ F.map _) _ =
    (F.map _ ≫ F.map _) _ * (F.map _ ≫ F.map _) _
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.map f) y) ((F.map g) y')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₁) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₃) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq (HMul.hMul ((CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Is …
  -/
  simp_rw [← F.map_comp, h₁, h₂, h₃, F.map_comp]
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.map f) y) ((F.map g) y')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₁) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₃) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq (HMul.hMul ((CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Is …
  -/
  congr 1
  /-
    case h.e_a
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.map f) y) ((F.map g) y')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₁) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₃) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F.map f) (F.map β)) y) ((CategoryTh …
  -/
  change F.map _ (F.map _ _) = F.map _ (F.map _ _)
  /-
    case h.e_a
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    j₁ : J
    y : ↑(F.obj j₁)
    j₂ : J
    x : ↑(F.obj j₂)
    j₃ : J
    y' : ↑(F.obj j₃)
    l : J
    f : Quiver.Hom ⟨j₁, y⟩.fst l
    g : Quiver.Hom ⟨j₃, y'⟩.fst l
    hfg : Eq ((F.map f) y) ((F.map g) y')
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₁) s
    β : Quiver.Hom l s
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max j₂ j₃) s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq ((F.map β) ((F.map f) y)) ((F.map β) ((F.map g) y'))
  -/
  rw [hfg]
  /-
    🎉 no goals
  -/


/-- Multiplication in the colimit. See also `colimitMulAux`. -/
@[to_additive "Addition in the colimit. See also `colimitAddAux`."]
noncomputable instance colimitMul : Mul (M.{v, u} F) :=
{ mul := fun x y => by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      x y : MonCat.FilteredColimits.M F
      ⊢ MonCat.FilteredColimits.M F
    -/
    refine Quot.lift₂ (colimitMulAux F) ?_ ?_ x y
      /-
        case refine_1
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y : MonCat.FilteredColimits.M F
        ⊢ ∀ (a b₁ b₂ : Sigma fun j => ↑(F.obj j)), CategoryTheory.Limits.Types.Quot.Re …
      -/
    · intro x y y' h
      /-
        case refine_1
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ y✝ : MonCat.FilteredColimits.M F
        x y y' : Sigma fun j => ↑(F.obj j)
        h : CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget MonCat …
        ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F x y) (MonCat.FilteredColimits.co …
      -/
      apply colimitMulAux_eq_of_rel_right
      /-
        case refine_1.hyy'
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ y✝ : MonCat.FilteredColimits.M F
        x y y' : Sigma fun j => ↑(F.obj j)
        h : CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget MonCat …
        ⊢ CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory.forg …
      -/
      apply Types.FilteredColimit.rel_of_quot_rel
      /-
        case refine_1.hyy'.a
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ y✝ : MonCat.FilteredColimits.M F
        x y y' : Sigma fun j => ↑(F.obj j)
        h : CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget MonCat …
        ⊢ CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget MonCat)) …
      -/
      exact h
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y : MonCat.FilteredColimits.M F
        ⊢ ∀ (a₁ a₂ b : Sigma fun j => ↑(F.obj j)), CategoryTheory.Limits.Types.Quot.Re …
      -/
    · intro x x' y h
      /-
        case refine_2
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ y✝ : MonCat.FilteredColimits.M F
        x x' y : Sigma fun j => ↑(F.obj j)
        h : CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget MonCat …
        ⊢ Eq (MonCat.FilteredColimits.colimitMulAux F x y) (MonCat.FilteredColimits.co …
      -/
      apply colimitMulAux_eq_of_rel_left
      /-
        case refine_2.hxx'
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ y✝ : MonCat.FilteredColimits.M F
        x x' y : Sigma fun j => ↑(F.obj j)
        h : CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget MonCat …
        ⊢ CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory.forg …
      -/
      apply Types.FilteredColimit.rel_of_quot_rel
      /-
        case refine_2.hxx'.a
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ y✝ : MonCat.FilteredColimits.M F
        x x' y : Sigma fun j => ↑(F.obj j)
        h : CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget MonCat …
        ⊢ CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget MonCat)) …
      -/
      exact h }
      /-
        🎉 no goals
      -/


/-- Multiplication in the colimit is independent of the chosen "maximum" in the filtered category.
In particular, this lemma allows us to "unfold" the definition of the multiplication of `x` and `y`,
using a custom object `k` and morphisms `f : x.1 ⟶ k` and `g : y.1 ⟶ k`.
-/
@[to_additive
      "Addition in the colimit is independent of the chosen \"maximum\" in the filtered
      category. In particular, this lemma allows us to \"unfold\" the definition of the addition of
      `x` and `y`, using a custom object `k` and morphisms `f : x.1 ⟶ k` and `g : y.1 ⟶ k`."]
theorem colimit_mul_mk_eq (x y : Σ j, F.obj j) (k : J) (f : x.1 ⟶ k) (g : y.1 ⟶ k) :
    M.mk.{v, u} F x * M.mk F y = M.mk F ⟨k, F.map f x.2 * F.map g y.2⟩ := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    x y : Sigma fun j => ↑(F.obj j)
    k : J
    f : Quiver.Hom x.fst k
    g : Quiver.Hom y.fst k
    ⊢ Eq (HMul.hMul (MonCat.FilteredColimits.M.mk F x) (MonCat.FilteredColimits.M. …
  -/
  obtain ⟨j₁, x⟩ := x; obtain ⟨j₂, y⟩ := y
  obtain ⟨s, α, β, h₁, h₂⟩ := IsFiltered.bowtie (IsFiltered.leftToMax j₁ j₂) f
    (IsFiltered.rightToMax j₁ j₂) g
  /-
    case mk.mk.intro.intro.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    k j₁ : J
    x : ↑(F.obj j₁)
    f : Quiver.Hom ⟨j₁, x⟩.fst k
    j₂ : J
    y : ↑(F.obj j₂)
    g : Quiver.Hom ⟨j₂, y⟩.fst k
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom k s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq (HMul.hMul (MonCat.FilteredColimits.M.mk F ⟨j₁, x⟩) (MonCat.FilteredColim …
  -/
  refine M.mk_eq F _ _ ?_
  /-
    case mk.mk.intro.intro.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    k j₁ : J
    x : ↑(F.obj j₁)
    f : Quiver.Hom ⟨j₁, x⟩.fst k
    j₂ : J
    y : ↑(F.obj j₂)
    g : Quiver.Hom ⟨j₂, y⟩.fst k
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom k s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Exists fun k_1 => Exists fun f_1 => Exists fun g_1 => Eq ((F.map f_1) ⟨Categ …
  -/
  use s, α, β
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    k j₁ : J
    x : ↑(F.obj j₁)
    f : Quiver.Hom ⟨j₁, x⟩.fst k
    j₂ : J
    y : ↑(F.obj j₂)
    g : Quiver.Hom ⟨j₂, y⟩.fst k
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom k s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq ((F.map α) ⟨CategoryTheory.IsFiltered.max ⟨j₁, x⟩.fst ⟨j₂, y⟩.fst, HMul.h …
  -/
  dsimp
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    k j₁ : J
    x : ↑(F.obj j₁)
    f : Quiver.Hom ⟨j₁, x⟩.fst k
    j₂ : J
    y : ↑(F.obj j₂)
    g : Quiver.Hom ⟨j₂, y⟩.fst k
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom k s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq ((F.map α) (HMul.hMul ((F.map (CategoryTheory.IsFiltered.leftToMax j₁ j₂) …
  -/
  simp_rw [MonoidHom.map_mul]
  -- Porting note: Lean cannot seem to use lemmas from concrete categories directly
  change (F.map _ ≫ F.map _) _ * (F.map _ ≫ F.map _) _ =
    (F.map _ ≫ F.map _) _ * (F.map _ ≫ F.map _) _
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    F : CategoryTheory.Functor J MonCatMax
    inst✝ : CategoryTheory.IsFiltered J
    k j₁ : J
    x : ↑(F.obj j₁)
    f : Quiver.Hom ⟨j₁, x⟩.fst k
    j₂ : J
    y : ↑(F.obj j₂)
    g : Quiver.Hom ⟨j₂, y⟩.fst k
    s : J
    α : Quiver.Hom (CategoryTheory.IsFiltered.max j₁ j₂) s
    β : Quiver.Hom k s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq (HMul.hMul ((CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Is …
  -/
  simp_rw [← F.map_comp, h₁, h₂]
  /-
    🎉 no goals
  -/


@[to_additive]
noncomputable instance colimitMulOneClass : MulOneClass (M.{v, u} F) :=
  { colimitOne F,
    colimitMul F with
    one_mul := fun x => by
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x : MonCat.FilteredColimits.M F
        ⊢ Eq (HMul.hMul 1 x) x
      -/
      refine Quot.inductionOn x ?_
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x : MonCat.FilteredColimits.M F
        ⊢ ∀ (a : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j), Eq (HM …
      -/
      intro x
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ : MonCat.FilteredColimits.M F
        x : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j
        ⊢ Eq (HMul.hMul 1 (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel (F.comp (Cate …
      -/
      obtain ⟨j, x⟩ := x
      rw [colimit_one_eq F j, colimit_mul_mk_eq F ⟨j, 1⟩ ⟨j, x⟩ j (𝟙 j) (𝟙 j), MonoidHom.map_one,
        one_mul, F.map_id]
      -- Porting note: `id_apply` does not work here, but the two sides are def-eq
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ : MonCat.FilteredColimits.M F
        j : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j
        ⊢ Eq (MonCat.FilteredColimits.M.mk F ⟨j, (CategoryTheory.CategoryStruct.id (F. …
      -/
      rfl
      /-
        🎉 no goals
      -/
    mul_one := fun x => by
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x : MonCat.FilteredColimits.M F
        ⊢ Eq (HMul.hMul x 1) x
      -/
      refine Quot.inductionOn x ?_
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x : MonCat.FilteredColimits.M F
        ⊢ ∀ (a : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j), Eq (HM …
      -/
      intro x
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ : MonCat.FilteredColimits.M F
        x : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel (F.comp (Catego …
      -/
      obtain ⟨j, x⟩ := x
      rw [colimit_one_eq F j, colimit_mul_mk_eq F ⟨j, x⟩ ⟨j, 1⟩ j (𝟙 j) (𝟙 j), MonoidHom.map_one,
        mul_one, F.map_id]
      -- Porting note: `id_apply` does not work here, but the two sides are def-eq
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x✝ : MonCat.FilteredColimits.M F
        j : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j
        ⊢ Eq (MonCat.FilteredColimits.M.mk F ⟨j, (CategoryTheory.CategoryStruct.id (F. …
      -/
      rfl }
      /-
        🎉 no goals
      -/


@[to_additive]
noncomputable instance colimitMonoid : Monoid (M.{v, u} F) :=
  { colimitMulOneClass F with
    mul_assoc := fun x y z => by
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y z : MonCat.FilteredColimits.M F
        ⊢ Eq (HMul.hMul (HMul.hMul x y) z) (HMul.hMul x (HMul.hMul y z))
      -/
      refine Quot.induction_on₃ x y z ?_
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y z : MonCat.FilteredColimits.M F
        ⊢ ∀ (a b c : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j), Eq …
      -/
      clear x y z
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        ⊢ ∀ (a b c : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j), Eq …
      -/
      intro x y z
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y z : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j
        ⊢ Eq (HMul.hMul (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel (F.c …
      -/
      obtain ⟨j₁, x⟩ := x
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        y z : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j
        j₁ : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j₁
        ⊢ Eq (HMul.hMul (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel (F.c …
      -/
      obtain ⟨j₂, y⟩ := y
      /-
        case mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        z : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j
        j₁ : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j₁
        j₂ : J
        y : (F.comp (CategoryTheory.forget MonCat)).obj j₂
        ⊢ Eq (HMul.hMul (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel (F.c …
      -/
      obtain ⟨j₃, z⟩ := z
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j₁
        j₂ : J
        y : (F.comp (CategoryTheory.forget MonCat)).obj j₂
        j₃ : J
        z : (F.comp (CategoryTheory.forget MonCat)).obj j₃
        ⊢ Eq (HMul.hMul (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel (F.c …
      -/
      change M.mk F _ * M.mk F _ * M.mk F _ = M.mk F _ * M.mk F _
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j₁
        j₂ : J
        y : (F.comp (CategoryTheory.forget MonCat)).obj j₂
        j₃ : J
        z : (F.comp (CategoryTheory.forget MonCat)).obj j₃
        ⊢ Eq (HMul.hMul (HMul.hMul (MonCat.FilteredColimits.M.mk F ⟨j₁, x⟩) (MonCat.Fi …
      -/
      dsimp
      rw [colimit_mul_mk_eq F ⟨j₁, x⟩ ⟨j₂, y⟩ (IsFiltered.max j₁ (IsFiltered.max j₂ j₃))
          (IsFiltered.leftToMax j₁ (IsFiltered.max j₂ j₃))
          (IsFiltered.leftToMax j₂ j₃ ≫ IsFiltered.rightToMax _ _),
        colimit_mul_mk_eq F ⟨(IsFiltered.max j₁ (IsFiltered.max j₂ j₃)), _⟩ ⟨j₃, z⟩
          (IsFiltered.max j₁ (IsFiltered.max j₂ j₃)) (𝟙 _)
          (IsFiltered.rightToMax j₂ j₃ ≫ IsFiltered.rightToMax _ _),
        colimit_mul_mk_eq.{v, u} F ⟨j₁, x⟩ ⟨IsFiltered.max j₂ j₃, _⟩ _
          (IsFiltered.leftToMax _ _) (IsFiltered.rightToMax _ _)]
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j₁
        j₂ : J
        y : (F.comp (CategoryTheory.forget MonCat)).obj j₂
        j₃ : J
        z : (F.comp (CategoryTheory.forget MonCat)).obj j₃
        ⊢ Eq (MonCat.FilteredColimits.M.mk F ⟨CategoryTheory.IsFiltered.max j₁ (Catego …
      -/
      congr 2
      /-
        case mk.mk.mk.e_a.e_snd
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j₁
        j₂ : J
        y : (F.comp (CategoryTheory.forget MonCat)).obj j₂
        j₃ : J
        z : (F.comp (CategoryTheory.forget MonCat)).obj j₃
        ⊢ Eq (HMul.hMul ((F.map (CategoryTheory.CategoryStruct.id ⟨CategoryTheory.IsFi …
      -/
      dsimp only
      rw [F.map_id, show ∀ x, (𝟙 (F.obj (IsFiltered.max j₁ (IsFiltered.max j₂ j₃)))) x = x
        from fun _ => rfl, mul_assoc, MonoidHom.map_mul, F.map_comp, F.map_comp]
      /-
        case mk.mk.mk.e_a.e_snd
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J MonCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : (F.comp (CategoryTheory.forget MonCat)).obj j₁
        j₂ : J
        y : (F.comp (CategoryTheory.forget MonCat)).obj j₂
        j₃ : J
        z : (F.comp (CategoryTheory.forget MonCat)).obj j₃
        ⊢ Eq (HMul.hMul ((F.map (CategoryTheory.IsFiltered.leftToMax j₁ (CategoryTheor …
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- The bundled monoid giving the filtered colimit of a diagram. -/
@[to_additive
  "The bundled additive monoid giving the filtered colimit of a diagram."]
noncomputable def colimit : MonCat.{max v u} :=
  MonCat.of (M.{v, u} F)


/-- The monoid homomorphism from a given monoid in the diagram to the colimit monoid. -/
@[to_additive
      "The additive monoid homomorphism from a given additive monoid in the diagram to the
      colimit additive monoid."]
def coconeMorphism (j : J) : F.obj j ⟶ colimit F where
  toFun := (Types.TypeMax.colimitCocone.{v, max v u, v} (F ⋙ forget MonCat)).ι.app j
  map_one' := (colimit_one_eq F j).symm
  map_mul' x y := by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      j : J
      x y : ↑(F.obj j)
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.TypeMax.colimitCocone (F.comp (C …
    -/
    convert (colimit_mul_mk_eq F ⟨j, x⟩ ⟨j, y⟩ j (𝟙 j) (𝟙 j)).symm
    /-
      case h.e'_2.h
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      j : J
      x y : ↑(F.obj j)
      e_1✝ : Eq (↑(MonCat.FilteredColimits.colimit F)) (MonCat.FilteredColimits.M F)
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.TypeMax.colimitCocone (F.comp (C …
    -/
    rw [F.map_id]
    /-
      case h.e'_2.h
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      j : J
      x y : ↑(F.obj j)
      e_1✝ : Eq (↑(MonCat.FilteredColimits.colimit F)) (MonCat.FilteredColimits.M F)
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.TypeMax.colimitCocone (F.comp (C …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem cocone_naturality {j j' : J} (f : j ⟶ j') :
    F.map f ≫ coconeMorphism.{v, u} F j' = coconeMorphism F j :=
  MonoidHom.ext fun x =>
    congr_fun ((Types.TypeMax.colimitCocone (F ⋙ forget MonCat)).ι.naturality f) x


/-- The cocone over the proposed colimit monoid. -/
@[to_additive "The cocone over the proposed colimit additive monoid."]
noncomputable def colimitCocone : Cocone F where
  pt := colimit.{v, u} F
  ι := { app := coconeMorphism F }


/-- Given a cocone `t` of `F`, the induced monoid homomorphism from the colimit to the cocone point.
As a function, this is simply given by the induced map of the corresponding cocone in `Type`.
The only thing left to see is that it is a monoid homomorphism.
-/
@[to_additive
      "Given a cocone `t` of `F`, the induced additive monoid homomorphism from the colimit
      to the cocone point. As a function, this is simply given by the induced map of the
      corresponding cocone in `Type`. The only thing left to see is that it is an additive monoid
      homomorphism."]
def colimitDesc (t : Cocone F) : colimit.{v, u} F ⟶ t.pt where
  toFun := (Types.TypeMax.colimitCoconeIsColimit.{v, max v u, v} (F ⋙ forget MonCat)).desc
    ((forget MonCat).mapCocone t)
  map_one' := by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      ⊢ Eq ((CategoryTheory.Limits.Types.TypeMax.colimitCoconeIsColimit (F.comp (Cat …
    -/
    rw [colimit_one_eq F IsFiltered.nonempty.some]
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      ⊢ Eq ((CategoryTheory.Limits.Types.TypeMax.colimitCoconeIsColimit (F.comp (Cat …
    -/
    exact MonoidHom.map_one _
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      x y : ↑(MonCat.FilteredColimits.colimit F)
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.TypeMax.colimitCoconeIsColimit ( …
    -/
    refine Quot.induction_on₂ x y ?_
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      x y : ↑(MonCat.FilteredColimits.colimit F)
      ⊢ ∀ (a b : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j), Eq ( …
    -/
    clear x y
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      ⊢ ∀ (a b : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j), Eq ( …
    -/
    intro x y
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      x y : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.TypeMax.colimitCoconeIsColimit ( …
    -/
    obtain ⟨i, x⟩ := x
    /-
      case mk
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      y : Sigma fun j => (F.comp (CategoryTheory.forget MonCat)).obj j
      i : J
      x : (F.comp (CategoryTheory.forget MonCat)).obj i
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.TypeMax.colimitCoconeIsColimit ( …
    -/
    obtain ⟨j, y⟩ := y
    rw [colimit_mul_mk_eq F ⟨i, x⟩ ⟨j, y⟩ (max' i j) (IsFiltered.leftToMax i j)
      (IsFiltered.rightToMax i j)]
    /-
      case mk.mk
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      i : J
      x : (F.comp (CategoryTheory.forget MonCat)).obj i
      j : J
      y : (F.comp (CategoryTheory.forget MonCat)).obj j
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.TypeMax.colimitCoconeIsColimit ( …
    -/
    dsimp [Types.TypeMax.colimitCoconeIsColimit]
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case mk.mk
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      i : J
      x : (F.comp (CategoryTheory.forget MonCat)).obj i
      j : J
      y : (F.comp (CategoryTheory.forget MonCat)).obj j
      ⊢ Eq ((t.ι.app (CategoryTheory.IsFiltered.max i j)) (HMul.hMul ((F.map (Catego …
    -/
    erw [MonoidHom.map_mul]
    -- Porting note: `rw` can't see through coercion is actually forgetful functor,
    -- so can't rewrite `t.w_apply`
    /-
      case mk.mk
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      i : J
      x : (F.comp (CategoryTheory.forget MonCat)).obj i
      j : J
      y : (F.comp (CategoryTheory.forget MonCat)).obj j
      ⊢ Eq (HMul.hMul ((t.ι.app (CategoryTheory.IsFiltered.max i j)) ((F.map (Catego …
    -/
    congr 1 <;>
    /-
      case mk.mk.e_a
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J MonCatMax
      inst✝ : CategoryTheory.IsFiltered J
      t : CategoryTheory.Limits.Cocone F
      i : J
      x : (F.comp (CategoryTheory.forget MonCat)).obj i
      j : J
      y : (F.comp (CategoryTheory.forget MonCat)).obj j
      ⊢ Eq ((t.ι.app (CategoryTheory.IsFiltered.max i j)) ((F.map (CategoryTheory.Is …
    -/
    /-
      🎉 no goals
    -/
    exact t.w_apply _ _
    /-
      🎉 no goals
    -/


/-- The proposed colimit cocone is a colimit in `MonCat`. -/
@[to_additive "The proposed colimit cocone is a colimit in `AddMonCat`."]
def colimitCoconeIsColimit : IsColimit (colimitCocone.{v, u} F) where
  desc := colimitDesc.{v, u} F
  fac t j := MonoidHom.ext fun x => congr_fun ((Types.TypeMax.colimitCoconeIsColimit.{v, u}
    (F ⋙ forget MonCat)).fac ((forget MonCat).mapCocone t) j) x
  uniq t m h := MonoidHom.ext fun y => congr_fun
      ((Types.TypeMax.colimitCoconeIsColimit (F ⋙ forget MonCat)).uniq ((forget MonCat).mapCocone t)
        ((forget MonCat).map m)
        fun j => funext fun x => DFunLike.congr_fun (i := MonCat.instFunLike _ _) (h j) x) y


@[to_additive]
instance forget_preservesFilteredColimits :
    PreservesFilteredColimits (forget MonCat.{u}) :=
  ⟨fun J hJ1 _ => letI hJ1' : Category J := hJ1
    ⟨fun {F} => preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit.{u, u} F)
      (Types.TypeMax.colimitCoconeIsColimit (F ⋙ forget MonCat.{u}))⟩⟩

/-- The colimit of `F ⋙ forget₂ CommMonCat MonCat` in the category `MonCat`.
In the following, we will show that this has the structure of a _commutative_ monoid.
-/
@[to_additive
      "The colimit of `F ⋙ forget₂ AddCommMonCat AddMonCat` in the category `AddMonCat`. In the
      following, we will show that this has the structure of a _commutative_ additive monoid."]
noncomputable abbrev M : MonCat.{max v u} :=
  MonCat.FilteredColimits.colimit.{v, u} (F ⋙ forget₂ CommMonCat MonCat.{max v u})


@[to_additive]
noncomputable instance colimitCommMonoid : CommMonoid.{max v u} (M.{v, u} F) :=
  { (M.{v, u} F) with
    mul_comm := fun x y => by
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J CommMonCat
        x y : ↑(CommMonCat.FilteredColimits.M F)
        ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
      -/
      refine Quot.induction_on₂ x y ?_
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J CommMonCat
        x y : ↑(CommMonCat.FilteredColimits.M F)
        ⊢ ∀ (a b : Sigma fun j => ((F.comp (CategoryTheory.forget₂ CommMonCat MonCat)) …
      -/
      clear x y
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J CommMonCat
        ⊢ ∀ (a b : Sigma fun j => ((F.comp (CategoryTheory.forget₂ CommMonCat MonCat)) …
      -/
      intro x y
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J CommMonCat
        x y : Sigma fun j => ((F.comp (CategoryTheory.forget₂ CommMonCat MonCat)).comp …
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      let k := max' x.1 y.1
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J CommMonCat
        x y : Sigma fun j => ((F.comp (CategoryTheory.forget₂ CommMonCat MonCat)).comp …
        k : J := CategoryTheory.IsFiltered.max x.fst y.fst
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      let f := IsFiltered.leftToMax x.1 y.1
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J CommMonCat
        x y : Sigma fun j => ((F.comp (CategoryTheory.forget₂ CommMonCat MonCat)).comp …
        k : J := CategoryTheory.IsFiltered.max x.fst y.fst
        f : Quiver.Hom x.fst (CategoryTheory.IsFiltered.max x.fst y.fst) := CategoryTh …
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      let g := IsFiltered.rightToMax x.1 y.1
      rw [colimit_mul_mk_eq.{v, u} (F ⋙ forget₂ CommMonCat MonCat) x y k f g,
        colimit_mul_mk_eq.{v, u} (F ⋙ forget₂ CommMonCat MonCat) y x k g f]
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J CommMonCat
        x y : Sigma fun j => ((F.comp (CategoryTheory.forget₂ CommMonCat MonCat)).comp …
        k : J := CategoryTheory.IsFiltered.max x.fst y.fst
        f : Quiver.Hom x.fst (CategoryTheory.IsFiltered.max x.fst y.fst) := CategoryTh …
        g : Quiver.Hom y.fst (CategoryTheory.IsFiltered.max x.fst y.fst) := CategoryTh …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ CommMonCat  …
      -/
      dsimp
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J CommMonCat
        x y : Sigma fun j => ((F.comp (CategoryTheory.forget₂ CommMonCat MonCat)).comp …
        k : J := CategoryTheory.IsFiltered.max x.fst y.fst
        f : Quiver.Hom x.fst (CategoryTheory.IsFiltered.max x.fst y.fst) := CategoryTh …
        g : Quiver.Hom y.fst (CategoryTheory.IsFiltered.max x.fst y.fst) := CategoryTh …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ CommMonCat  …
      -/
      rw [mul_comm] }
      /-
        🎉 no goals
      -/


/-- The bundled commutative monoid giving the filtered colimit of a diagram. -/
@[to_additive "The bundled additive commutative monoid giving the filtered colimit of a diagram."]
noncomputable def colimit : CommMonCat.{max v u} :=
  CommMonCat.of (M.{v, u} F)


/-- The cocone over the proposed colimit commutative monoid. -/
@[to_additive "The cocone over the proposed colimit additive commutative monoid."]
noncomputable def colimitCocone : Cocone F where
  pt := colimit.{v, u} F
  ι := { (MonCat.FilteredColimits.colimitCocone.{v, u}
    (F ⋙ forget₂ CommMonCat MonCat.{max v u})).ι with }


/-- The proposed colimit cocone is a colimit in `CommMonCat`. -/
@[to_additive "The proposed colimit cocone is a colimit in `AddCommMonCat`."]
def colimitCoconeIsColimit : IsColimit (colimitCocone.{v, u} F) where
  desc t :=
    MonCat.FilteredColimits.colimitDesc.{v, u} (F ⋙ forget₂ CommMonCat MonCat.{max v u})
      ((forget₂ CommMonCat MonCat.{max v u}).mapCocone t)
  fac t j :=
    DFunLike.coe_injective (i := CommMonCat.instFunLike _ _) <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget CommMonCat.{max v u})).fac
        ((forget CommMonCat).mapCocone t) j
  uniq t m h :=
    DFunLike.coe_injective (i := CommMonCat.instFunLike _ _) <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget CommMonCat.{max v u})).uniq
        ((forget CommMonCat.{max v u}).mapCocone t)
        ((forget CommMonCat.{max v u}).map m) fun j => funext fun x =>
          DFunLike.congr_fun (i := CommMonCat.instFunLike _ _) (h j) x


@[to_additive forget₂AddMonPreservesFilteredColimits]
noncomputable instance forget₂Mon_preservesFilteredColimits :
  PreservesFilteredColimits (forget₂ CommMonCat MonCat.{u}) :=
⟨fun J hJ1 _ => letI hJ3 : Category J := hJ1
  ⟨fun {F} => preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit.{u, u} F)
    (MonCat.FilteredColimits.colimitCoconeIsColimit (F ⋙ forget₂ CommMonCat MonCat.{u}))⟩⟩


@[to_additive]
noncomputable instance forget_preservesFilteredColimits :
    PreservesFilteredColimits (forget CommMonCat.{u}) :=
  Limits.comp_preservesFilteredColimits (forget₂ CommMonCat MonCat) (forget MonCat)


