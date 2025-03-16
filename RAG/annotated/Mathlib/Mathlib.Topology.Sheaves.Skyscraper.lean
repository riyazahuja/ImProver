/-- A skyscraper presheaf is a presheaf supported at a single point: if `p₀ ∈ X` is a specified
point, then the skyscraper presheaf `𝓕` with value `A` is defined by `U ↦ A` if `p₀ ∈ U` and
`U ↦ *` if `p₀ ∉ A` where `*` is some terminal object.
-/
@[simps]
def skyscraperPresheaf : Presheaf C X where
  obj U := if p₀ ∈ unop U then A else terminal C
  map {U V} i :=
                                          /-
                                            X : TopCat
                                            p₀ : ↑X
                                            inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
                                            C : Type v
                                            inst✝¹ : CategoryTheory.Category.{w, v} C
                                            inst✝ : CategoryTheory.Limits.HasTerminal C
                                            A : C
                                            U V : Opposite (TopologicalSpace.Opens ↑X)
                                            i : Quiver.Hom U V
                                            h : Membership.mem (Opposite.unop V) p₀
                                            ⊢ Eq ((fun U => ite (Membership.mem (Opposite.unop U) p₀) A (CategoryTheory.Li …
                                          -/
    if h : p₀ ∈ unop V then eqToHom <| by dsimp; rw [if_pos h, if_pos (by simpa using i.unop.le h)]
                                                 /-
                                                   🎉 no goals
                                                 -/
    else ((if_neg h).symm.ndrec terminalIsTerminal).from _
  map_id U :=
    (em (p₀ ∈ U.unop)).elim (fun h => dif_pos h) fun h =>
      ((if_neg h).symm.ndrec terminalIsTerminal).hom_ext _ _
  map_comp {U V W} iVU iWV := by
    /-
      X : TopCat
      p₀ : ↑X
      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} C
      inst✝ : CategoryTheory.Limits.HasTerminal C
      A : C
      U V W : Opposite (TopologicalSpace.Opens ↑X)
      iVU : Quiver.Hom U V
      iWV : Quiver.Hom V W
      ⊢ Eq ({ obj := fun U => ite (Membership.mem (Opposite.unop U) p₀) A (CategoryT …
    -/
    by_cases hW : p₀ ∈ unop W
      /-
        case pos
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        A : C
        U V W : Opposite (TopologicalSpace.Opens ↑X)
        iVU : Quiver.Hom U V
        iWV : Quiver.Hom V W
        hW : Membership.mem (Opposite.unop W) p₀
        ⊢ Eq ({ obj := fun U => ite (Membership.mem (Opposite.unop U) p₀) A (CategoryT …
      -/
    · have hV : p₀ ∈ unop V := leOfHom iWV.unop hW
      /-
        case pos
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        A : C
        U V W : Opposite (TopologicalSpace.Opens ↑X)
        iVU : Quiver.Hom U V
        iWV : Quiver.Hom V W
        hW : Membership.mem (Opposite.unop W) p₀
        hV : Membership.mem (Opposite.unop V) p₀
        ⊢ Eq ({ obj := fun U => ite (Membership.mem (Opposite.unop U) p₀) A (CategoryT …
      -/
      simp only [dif_pos hW, dif_pos hV, eqToHom_trans]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        A : C
        U V W : Opposite (TopologicalSpace.Opens ↑X)
        iVU : Quiver.Hom U V
        iWV : Quiver.Hom V W
        hW : Not (Membership.mem (Opposite.unop W) p₀)
        ⊢ Eq ({ obj := fun U => ite (Membership.mem (Opposite.unop U) p₀) A (CategoryT …
      -/
    · dsimp; rw [dif_neg hW]; apply ((if_neg hW).symm.ndrec terminalIsTerminal).hom_ext
                              /-
                                🎉 no goals
                              -/


theorem skyscraperPresheaf_eq_pushforward
    [hd : ∀ U : Opens (TopCat.of PUnit.{u + 1}), Decidable (PUnit.unit ∈ U)] :
    skyscraperPresheaf p₀ A =
      ContinuousMap.const (TopCat.of PUnit) p₀ _*
        skyscraperPresheaf (X := TopCat.of PUnit) PUnit.unit A := by
  convert_to @skyscraperPresheaf X p₀ (fun U => hd <| (Opens.map <| ContinuousMap.const _ p₀).obj U)
                    /-
                      case h.e'_2
                      X : TopCat
                      p₀ : ↑X
                      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
                      C : Type v
                      inst✝¹ : CategoryTheory.Category.{w, v} C
                      inst✝ : CategoryTheory.Limits.HasTerminal C
                      A : C
                      hd : (U : TopologicalSpace.Opens ↑(TopCat.of PUnit.{u + 1})) → Decidable (Memb …
                      ⊢ Eq (skyscraperPresheaf p₀ A) (skyscraperPresheaf p₀ A)
                    -/
                    /-
                      🎉 no goals
                    -/
    C _ _ A = _ <;> congr
                    /-
                      🎉 no goals
                    -/


/-- Taking skyscraper presheaf at a point is functorial: `c ↦ skyscraper p₀ c` defines a functor by
sending every `f : a ⟶ b` to the natural transformation `α` defined as: `α(U) = f : a ⟶ b` if
`p₀ ∈ U` and the unique morphism to a terminal object in `C` if `p₀ ∉ U`.
-/
@[simps]
def SkyscraperPresheafFunctor.map' {a b : C} (f : a ⟶ b) :
    skyscraperPresheaf p₀ a ⟶ skyscraperPresheaf p₀ b where
  app U :=
    if h : p₀ ∈ U.unop then eqToHom (if_pos h) ≫ f ≫ eqToHom (if_pos h).symm
    else ((if_neg h).symm.ndrec terminalIsTerminal).from _
  naturality U V i := by
    /-
      X : TopCat
      p₀ : ↑X
      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} C
      inst✝ : CategoryTheory.Limits.HasTerminal C
      A a b : C
      f : Quiver.Hom a b
      U V : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((skyscraperPresheaf p₀ a).map i) ((f …
    -/
    simp only [skyscraperPresheaf_map]
    /-
      X : TopCat
      p₀ : ↑X
      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} C
      inst✝ : CategoryTheory.Limits.HasTerminal C
      A a b : C
      f : Quiver.Hom a b
      U V : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Membership.mem (Opposite.unop  …
    -/
    by_cases hV : p₀ ∈ V.unop
      /-
        case pos
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        A a b : C
        f : Quiver.Hom a b
        U V : Opposite (TopologicalSpace.Opens ↑X)
        i : Quiver.Hom U V
        hV : Membership.mem (Opposite.unop V) p₀
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Membership.mem (Opposite.unop  …
      -/
    · have hU : p₀ ∈ U.unop := leOfHom i.unop hV
      simp only [skyscraperPresheaf_obj, hU, hV, ↓reduceDIte, eqToHom_trans_assoc, Category.assoc,
        eqToHom_trans]
      /-
        case neg
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        A a b : C
        f : Quiver.Hom a b
        U V : Opposite (TopologicalSpace.Opens ↑X)
        i : Quiver.Hom U V
        hV : Not (Membership.mem (Opposite.unop V) p₀)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Membership.mem (Opposite.unop  …
      -/
    · apply ((if_neg hV).symm.ndrec terminalIsTerminal).hom_ext
      /-
        🎉 no goals
      -/


theorem SkyscraperPresheafFunctor.map'_id {a : C} :
    SkyscraperPresheafFunctor.map' p₀ (𝟙 a) = 𝟙 _ := by
  /-
    X : TopCat
    p₀ : ↑X
    inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    a : C
    ⊢ Eq (SkyscraperPresheafFunctor.map' p₀ (CategoryTheory.CategoryStruct.id a))  …
  -/
  ext U
  /-
    case w
    X : TopCat
    p₀ : ↑X
    inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    a : C
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq ((SkyscraperPresheafFunctor.map' p₀ (CategoryTheory.CategoryStruct.id a)) …
  -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  simp only [SkyscraperPresheafFunctor.map'_app, NatTrans.id_app]; split_ifs <;> aesop_cat
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem SkyscraperPresheafFunctor.map'_comp {a b c : C} (f : a ⟶ b) (g : b ⟶ c) :
    SkyscraperPresheafFunctor.map' p₀ (f ≫ g) =
      SkyscraperPresheafFunctor.map' p₀ f ≫ SkyscraperPresheafFunctor.map' p₀ g := by
  /-
    X : TopCat
    p₀ : ↑X
    inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    a b c : C
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    ⊢ Eq (SkyscraperPresheafFunctor.map' p₀ (CategoryTheory.CategoryStruct.comp f  …
  -/
  ext U
  /-
    case w
    X : TopCat
    p₀ : ↑X
    inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    a b c : C
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq ((SkyscraperPresheafFunctor.map' p₀ (CategoryTheory.CategoryStruct.comp f …
  -/
  simp only [SkyscraperPresheafFunctor.map'_app, NatTrans.comp_app]
  /-
    case w
    X : TopCat
    p₀ : ↑X
    inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    a b c : C
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq (dite (Membership.mem U p₀) (fun h => CategoryTheory.CategoryStruct.comp  …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> aesop_cat
                       /-
                         🎉 no goals
                       -/


/-- Taking skyscraper presheaf at a point is functorial: `c ↦ skyscraper p₀ c` defines a functor by
sending every `f : a ⟶ b` to the natural transformation `α` defined as: `α(U) = f : a ⟶ b` if
`p₀ ∈ U` and the unique morphism to a terminal object in `C` if `p₀ ∉ U`.
-/
@[simps]
def skyscraperPresheafFunctor : C ⥤ Presheaf C X where
  obj := skyscraperPresheaf p₀
  map := SkyscraperPresheafFunctor.map' p₀
  map_id _ := SkyscraperPresheafFunctor.map'_id p₀
  map_comp := SkyscraperPresheafFunctor.map'_comp p₀


/-- The cocone at `A` for the stalk functor of `skyscraperPresheaf p₀ A` when `y ∈ closure {p₀}`
-/
@[simps]
def skyscraperPresheafCoconeOfSpecializes {y : X} (h : p₀ ⤳ y) :
    Cocone ((OpenNhds.inclusion y).op ⋙ skyscraperPresheaf p₀ A) where
  pt := A
  ι :=
    { app := fun U => eqToHom <| if_pos <| h.mem_open U.unop.1.2 U.unop.2
      naturality := fun U V inc => by
        /-
          X : TopCat
          p₀ : ↑X
          inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
          C : Type v
          inst✝¹ : CategoryTheory.Category.{u, v} C
          A : C
          inst✝ : CategoryTheory.Limits.HasTerminal C
          y : ↑X
          h : Specializes p₀ y
          U V : Opposite (TopologicalSpace.OpenNhds y)
          inc : Quiver.Hom U V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((TopologicalSpace.OpenNhds.inclusio …
        -/
        change dite _ _ _ ≫ _ = _; rw [dif_pos]
        /-
          X : TopCat
          p₀ : ↑X
          inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
          C : Type v
          inst✝¹ : CategoryTheory.Category.{u, v} C
          A : C
          inst✝ : CategoryTheory.Limits.HasTerminal C
          y : ↑X
          h : Specializes p₀ y
          U V : Opposite (TopologicalSpace.OpenNhds y)
          inc : Quiver.Hom U V
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ((fun U => …
        -/
        swap -- Porting note: swap goal to prevent proving same thing twice
          /-
            case hc
            X : TopCat
            p₀ : ↑X
            inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
            C : Type v
            inst✝¹ : CategoryTheory.Category.{u, v} C
            A : C
            inst✝ : CategoryTheory.Limits.HasTerminal C
            y : ↑X
            h : Specializes p₀ y
            U V : Opposite (TopologicalSpace.OpenNhds y)
            inc : Quiver.Hom U V
            ⊢ Membership.mem (Opposite.unop ((TopologicalSpace.OpenNhds.inclusion y).op.ob …
          -/
        · exact h.mem_open V.unop.1.2 V.unop.2
          /-
            🎉 no goals
          -/
        · simp only [Functor.comp_obj, Functor.op_obj, skyscraperPresheaf_obj, unop_op,
            Functor.const_obj_obj, eqToHom_trans, Functor.const_obj_map, Category.comp_id] }


/--
The cocone at `A` for the stalk functor of `skyscraperPresheaf p₀ A` when `y ∈ closure {p₀}` is a
colimit
-/
noncomputable def skyscraperPresheafCoconeIsColimitOfSpecializes {y : X} (h : p₀ ⤳ y) :
    IsColimit (skyscraperPresheafCoconeOfSpecializes p₀ A h) where
  desc c := eqToHom (if_pos trivial).symm ≫ c.ι.app (op ⊤)
  fac c U := by
    /-
      X : TopCat
      p₀ : ↑X
      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝¹ : CategoryTheory.Category.{u, v} C
      A : C
      inst✝ : CategoryTheory.Limits.HasTerminal C
      y : ↑X
      h : Specializes p₀ y
      c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
      U : Opposite (TopologicalSpace.OpenNhds y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((skyscraperPresheafCoconeOfSpecializ …
    -/
    dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):added a `dsimp`
    /-
      X : TopCat
      p₀ : ↑X
      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝¹ : CategoryTheory.Category.{u, v} C
      A : C
      inst✝ : CategoryTheory.Limits.HasTerminal C
      y : ↑X
      h : Specializes p₀ y
      c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
      U : Opposite (TopologicalSpace.OpenNhds y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    rw [← c.w (homOfLE <| (le_top : unop U ≤ _)).op]
    /-
      X : TopCat
      p₀ : ↑X
      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝¹ : CategoryTheory.Category.{u, v} C
      A : C
      inst✝ : CategoryTheory.Limits.HasTerminal C
      y : ↑X
      h : Specializes p₀ y
      c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
      U : Opposite (TopologicalSpace.OpenNhds y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    change _ ≫ _ ≫ dite _ _ _ ≫ _ = _
    /-
      X : TopCat
      p₀ : ↑X
      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝¹ : CategoryTheory.Category.{u, v} C
      A : C
      inst✝ : CategoryTheory.Limits.HasTerminal C
      y : ↑X
      h : Specializes p₀ y
      c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
      U : Opposite (TopologicalSpace.OpenNhds y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
    -/
    rw [dif_pos]
    · simp only [skyscraperPresheafCoconeOfSpecializes_ι_app, eqToHom_trans_assoc,
        eqToHom_refl, Category.id_comp, unop_op, op_unop]
      /-
        case hc
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{u, v} C
        A : C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        y : ↑X
        h : Specializes p₀ y
        c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
        U : Opposite (TopologicalSpace.OpenNhds y)
        ⊢ Membership.mem (Opposite.unop ((TopologicalSpace.OpenNhds.inclusion y).op.ob …
      -/
    · exact h.mem_open U.unop.1.2 U.unop.2
      /-
        🎉 no goals
      -/
  uniq c f h := by
    /-
      X : TopCat
      p₀ : ↑X
      inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝¹ : CategoryTheory.Category.{u, v} C
      A : C
      inst✝ : CategoryTheory.Limits.HasTerminal C
      y : ↑X
      h✝ : Specializes p₀ y
      c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
      f : Quiver.Hom (skyscraperPresheafCoconeOfSpecializes p₀ A h✝).pt c.pt
      h : ∀ (j : Opposite (TopologicalSpace.OpenNhds y)), Eq (CategoryTheory.Categor …
      ⊢ Eq f ((fun c => CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
    -/
    dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):added a `dsimp`
    rw [← h, skyscraperPresheafCoconeOfSpecializes_ι_app, eqToHom_trans_assoc, eqToHom_refl,
      Category.id_comp]


/-- If `y ∈ closure {p₀}`, then the stalk of `skyscraperPresheaf p₀ A` at `y` is `A`.
-/
noncomputable def skyscraperPresheafStalkOfSpecializes [HasColimits C] {y : X} (h : p₀ ⤳ y) :
    (skyscraperPresheaf p₀ A).stalk y ≅ A :=
  colimit.isoColimitCocone ⟨_, skyscraperPresheafCoconeIsColimitOfSpecializes p₀ A h⟩


@[reassoc (attr := simp)]
lemma germ_skyscraperPresheafStalkOfSpecializes_hom [HasColimits C] {y : X} (h : p₀ ⤳ y) (U hU) :
    (skyscraperPresheaf p₀ A).germ U y hU ≫
      (skyscraperPresheafStalkOfSpecializes p₀ A h).hom = eqToHom (if_pos (h.mem_open U.2 hU)) :=
  colimit.isoColimitCocone_ι_hom _ _


/-- The cocone at `*` for the stalk functor of `skyscraperPresheaf p₀ A` when `y ∉ closure {p₀}`
-/
@[simps]
def skyscraperPresheafCocone (y : X) :
    Cocone ((OpenNhds.inclusion y).op ⋙ skyscraperPresheaf p₀ A) where
  pt := terminal C
  ι :=
    { app := fun _ => terminal.from _
      naturality := fun _ _ _ => terminalIsTerminal.hom_ext _ _ }


/--
The cocone at `*` for the stalk functor of `skyscraperPresheaf p₀ A` when `y ∉ closure {p₀}` is a
colimit
-/
noncomputable def skyscraperPresheafCoconeIsColimitOfNotSpecializes {y : X} (h : ¬p₀ ⤳ y) :
    IsColimit (skyscraperPresheafCocone p₀ A y) :=
  let h1 : ∃ U : OpenNhds y, p₀ ∉ U.1 :=
    let ⟨U, ho, h₀, hy⟩ := not_specializes_iff_exists_open.mp h
    ⟨⟨⟨U, ho⟩, h₀⟩, hy⟩
  { desc := fun c => eqToHom (if_neg h1.choose_spec).symm ≫ c.ι.app (op h1.choose)
    fac := fun c U => by
      /-
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{u, v} C
        A : C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        y : ↑X
        h : Not (Specializes p₀ y)
        h1 : Exists fun U => Not (Membership.mem U.obj p₀) := skyscraperPresheafCocone …
        c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
        U : Opposite (TopologicalSpace.OpenNhds y)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((skyscraperPresheafCocone p₀ A y).ι. …
      -/
      change _ = c.ι.app (op U.unop)
      simp only [← c.w (homOfLE <| @inf_le_left _ _ h1.choose U.unop).op, ←
        c.w (homOfLE <| @inf_le_right _ _ h1.choose U.unop).op, ← Category.assoc]
      /-
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{u, v} C
        A : C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        y : ↑X
        h : Not (Specializes p₀ y)
        h1 : Exists fun U => Not (Membership.mem U.obj p₀) := skyscraperPresheafCocone …
        c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
        U : Opposite (TopologicalSpace.OpenNhds y)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      congr 1
      /-
        case e_a
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{u, v} C
        A : C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        y : ↑X
        h : Not (Specializes p₀ y)
        h1 : Exists fun U => Not (Membership.mem U.obj p₀) := skyscraperPresheafCocone …
        c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
        U : Opposite (TopologicalSpace.OpenNhds y)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      refine ((if_neg ?_).symm.ndrec terminalIsTerminal).hom_ext _ _
      /-
        case e_a
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{u, v} C
        A : C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        y : ↑X
        h : Not (Specializes p₀ y)
        h1 : Exists fun U => Not (Membership.mem U.obj p₀) := skyscraperPresheafCocone …
        c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
        U : Opposite (TopologicalSpace.OpenNhds y)
        ⊢ Not (Membership.mem (Opposite.unop ((TopologicalSpace.OpenNhds.inclusion y). …
      -/
      exact fun h => h1.choose_spec h.1
      /-
        🎉 no goals
      -/
    uniq := fun c f H => by
      /-
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{u, v} C
        A : C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        y : ↑X
        h : Not (Specializes p₀ y)
        h1 : Exists fun U => Not (Membership.mem U.obj p₀) := skyscraperPresheafCocone …
        c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
        f : Quiver.Hom (skyscraperPresheafCocone p₀ A y).pt c.pt
        H : ∀ (j : Opposite (TopologicalSpace.OpenNhds y)), Eq (CategoryTheory.Categor …
        ⊢ Eq f ((fun c => CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
      -/
      dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):added a `dsimp`
      /-
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{u, v} C
        A : C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        y : ↑X
        h : Not (Specializes p₀ y)
        h1 : Exists fun U => Not (Membership.mem U.obj p₀) := skyscraperPresheafCocone …
        c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
        f : Quiver.Hom (skyscraperPresheafCocone p₀ A y).pt c.pt
        H : ∀ (j : Opposite (TopologicalSpace.OpenNhds y)), Eq (CategoryTheory.Categor …
        ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (c.ι.app …
      -/
      rw [← Category.id_comp f, ← H, ← Category.assoc]
      /-
        X : TopCat
        p₀ : ↑X
        inst✝² : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝¹ : CategoryTheory.Category.{u, v} C
        A : C
        inst✝ : CategoryTheory.Limits.HasTerminal C
        y : ↑X
        h : Not (Specializes p₀ y)
        h1 : Exists fun U => Not (Membership.mem U.obj p₀) := skyscraperPresheafCocone …
        c : CategoryTheory.Limits.Cocone ((TopologicalSpace.OpenNhds.inclusion y).op.c …
        f : Quiver.Hom (skyscraperPresheafCocone p₀ A y).pt c.pt
        H : ∀ (j : Opposite (TopologicalSpace.OpenNhds y)), Eq (CategoryTheory.Categor …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (sk …
      -/
      congr 1; apply terminalIsTerminal.hom_ext }
               /-
                 🎉 no goals
               -/


/-- If `y ∉ closure {p₀}`, then the stalk of `skyscraperPresheaf p₀ A` at `y` is isomorphic to a
terminal object.
-/
noncomputable def skyscraperPresheafStalkOfNotSpecializes [HasColimits C] {y : X} (h : ¬p₀ ⤳ y) :
    (skyscraperPresheaf p₀ A).stalk y ≅ terminal C :=
  colimit.isoColimitCocone ⟨_, skyscraperPresheafCoconeIsColimitOfNotSpecializes _ A h⟩


/-- If `y ∉ closure {p₀}`, then the stalk of `skyscraperPresheaf p₀ A` at `y` is a terminal object
-/
def skyscraperPresheafStalkOfNotSpecializesIsTerminal [HasColimits C] {y : X} (h : ¬p₀ ⤳ y) :
    IsTerminal ((skyscraperPresheaf p₀ A).stalk y) :=
  IsTerminal.ofIso terminalIsTerminal <| (skyscraperPresheafStalkOfNotSpecializes _ _ h).symm


theorem skyscraperPresheaf_isSheaf : (skyscraperPresheaf p₀ A).IsSheaf := by
  classical exact
    (Presheaf.isSheaf_iso_iff (eqToIso <| skyscraperPresheaf_eq_pushforward p₀ A)).mpr <|
      (Sheaf.pushforward_sheaf_of_sheaf _
        (Presheaf.isSheaf_on_punit_of_isTerminal _ (by
          dsimp [skyscraperPresheaf]
          rw [if_neg]
          · exact terminalIsTerminal
          · #adaptation_note /-- 2024-03-24
            Previously the universe annotation was not needed here. -/
            exact Set.not_mem_empty PUnit.unit.{u+1})))


/--
The skyscraper presheaf supported at `p₀` with value `A` is the sheaf that assigns `A` to all opens
`U` that contain `p₀` and assigns `*` otherwise.
-/
def skyscraperSheaf : Sheaf C X :=
  ⟨skyscraperPresheaf p₀ A, skyscraperPresheaf_isSheaf _ _⟩


/-- Taking skyscraper sheaf at a point is functorial: `c ↦ skyscraper p₀ c` defines a functor by
sending every `f : a ⟶ b` to the natural transformation `α` defined as: `α(U) = f : a ⟶ b` if
`p₀ ∈ U` and the unique morphism to a terminal object in `C` if `p₀ ∉ U`.
-/
def skyscraperSheafFunctor : C ⥤ Sheaf C X where
  obj c := skyscraperSheaf p₀ c
  map f := Sheaf.Hom.mk <| (skyscraperPresheafFunctor p₀).map f
  map_id _ := Sheaf.Hom.ext <| (skyscraperPresheafFunctor p₀).map_id _
  map_comp _ _ := Sheaf.Hom.ext <| (skyscraperPresheafFunctor p₀).map_comp _ _


/-- If `f : 𝓕.stalk p₀ ⟶ c`, then a natural transformation `𝓕 ⟶ skyscraperPresheaf p₀ c` can be
defined by: `𝓕.germ p₀ ≫ f : 𝓕(U) ⟶ c` if `p₀ ∈ U` and the unique morphism to a terminal object
if `p₀ ∉ U`.
-/
@[simps]
def toSkyscraperPresheaf {𝓕 : Presheaf C X} {c : C} (f : 𝓕.stalk p₀ ⟶ c) :
    𝓕 ⟶ skyscraperPresheaf p₀ c where
  app U :=
    if h : p₀ ∈ U.unop then 𝓕.germ _ p₀ h ≫ f ≫ eqToHom (if_pos h).symm
    else ((if_neg h).symm.ndrec terminalIsTerminal).from _
  naturality U V inc := by
    -- Porting note: don't know why original proof fell short of working, add `aesop_cat` finished
    -- the proofs anyway
    /-
      X : TopCat
      p₀ : ↑X
      inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      𝓕 : TopCat.Presheaf C X
      c : C
      f : Quiver.Hom (𝓕.stalk p₀) c
      U V : Opposite (TopologicalSpace.Opens ↑X)
      inc : Quiver.Hom U V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓕.map inc) ((fun U => dite (Membersh …
    -/
    dsimp
    /-
      X : TopCat
      p₀ : ↑X
      inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      𝓕 : TopCat.Presheaf C X
      c : C
      f : Quiver.Hom (𝓕.stalk p₀) c
      U V : Opposite (TopologicalSpace.Opens ↑X)
      inc : Quiver.Hom U V
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓕.map inc) (dite (Membership.mem (Op …
    -/
    by_cases hV : p₀ ∈ V.unop
      /-
        case pos
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        𝓕 : TopCat.Presheaf C X
        c : C
        f : Quiver.Hom (𝓕.stalk p₀) c
        U V : Opposite (TopologicalSpace.Opens ↑X)
        inc : Quiver.Hom U V
        hV : Membership.mem (Opposite.unop V) p₀
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓕.map inc) (dite (Membership.mem (Op …
      -/
    · have hU : p₀ ∈ U.unop := leOfHom inc.unop hV
      /-
        case pos
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        𝓕 : TopCat.Presheaf C X
        c : C
        f : Quiver.Hom (𝓕.stalk p₀) c
        U V : Opposite (TopologicalSpace.Opens ↑X)
        inc : Quiver.Hom U V
        hV : Membership.mem (Opposite.unop V) p₀
        hU : Membership.mem (Opposite.unop U) p₀
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓕.map inc) (dite (Membership.mem (Op …
      -/
      split_ifs
      /-
        case pos
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        𝓕 : TopCat.Presheaf C X
        c : C
        f : Quiver.Hom (𝓕.stalk p₀) c
        U V : Opposite (TopologicalSpace.Opens ↑X)
        inc : Quiver.Hom U V
        hV : Membership.mem (Opposite.unop V) p₀
        hU : Membership.mem (Opposite.unop U) p₀
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓕.map inc) (CategoryTheory.CategoryS …
      -/
      rw [← Category.assoc, 𝓕.germ_res' inc, Category.assoc, Category.assoc, eqToHom_trans]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        𝓕 : TopCat.Presheaf C X
        c : C
        f : Quiver.Hom (𝓕.stalk p₀) c
        U V : Opposite (TopologicalSpace.Opens ↑X)
        inc : Quiver.Hom U V
        hV : Not (Membership.mem (Opposite.unop V) p₀)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓕.map inc) (dite (Membership.mem (Op …
      -/
    · split_ifs
      /-
        case neg
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        𝓕 : TopCat.Presheaf C X
        c : C
        f : Quiver.Hom (𝓕.stalk p₀) c
        U V : Opposite (TopologicalSpace.Opens ↑X)
        inc : Quiver.Hom U V
        hV : Not (Membership.mem (Opposite.unop V) p₀)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓕.map inc) ((Eq.rec CategoryTheory.L …
      -/
      exact ((if_neg hV).symm.ndrec terminalIsTerminal).hom_ext ..
      /-
        🎉 no goals
      -/


/-- If `f : 𝓕 ⟶ skyscraperPresheaf p₀ c` is a natural transformation, then there is a morphism
`𝓕.stalk p₀ ⟶ c` defined as the morphism from colimit to cocone at `c`.
-/
def fromStalk {𝓕 : Presheaf C X} {c : C} (f : 𝓕 ⟶ skyscraperPresheaf p₀ c) : 𝓕.stalk p₀ ⟶ c :=
  let χ : Cocone ((OpenNhds.inclusion p₀).op ⋙ 𝓕) :=
    Cocone.mk c <|
      { app := fun U => f.app ((OpenNhds.inclusion p₀).op.obj U) ≫ eqToHom (if_pos U.unop.2)
        naturality := fun U V inc => by
          dsimp only [Functor.const_obj_map, Functor.const_obj_obj, Functor.comp_map,
            Functor.comp_obj, Functor.op_obj, skyscraperPresheaf_obj]
          rw [Category.comp_id, ← Category.assoc, comp_eqToHom_iff, Category.assoc,
            eqToHom_trans, f.naturality, skyscraperPresheaf_map]
          /-
            X : TopCat
            p₀ : ↑X
            inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
            C : Type v
            inst✝² : CategoryTheory.Category.{u, v} C
            A : C
            inst✝¹ : CategoryTheory.Limits.HasTerminal C
            inst✝ : CategoryTheory.Limits.HasColimits C
            𝓕 : TopCat.Presheaf C X
            c : C
            f : Quiver.Hom 𝓕 (skyscraperPresheaf p₀ c)
            U V : Opposite (TopologicalSpace.OpenNhds p₀)
            inc : Quiver.Hom U V
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := (TopologicalSpace.Op …
          -/
          have hV : p₀ ∈ (OpenNhds.inclusion p₀).obj V.unop := V.unop.2
          /-
            X : TopCat
            p₀ : ↑X
            inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
            C : Type v
            inst✝² : CategoryTheory.Category.{u, v} C
            A : C
            inst✝¹ : CategoryTheory.Limits.HasTerminal C
            inst✝ : CategoryTheory.Limits.HasColimits C
            𝓕 : TopCat.Presheaf C X
            c : C
            f : Quiver.Hom 𝓕 (skyscraperPresheaf p₀ c)
            U V : Opposite (TopologicalSpace.OpenNhds p₀)
            inc : Quiver.Hom U V
            hV : Membership.mem ((TopologicalSpace.OpenNhds.inclusion p₀).obj (Opposite.un …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := (TopologicalSpace.Op …
          -/
          simp only [dif_pos hV] }
          /-
            🎉 no goals
          -/
  colimit.desc _ χ


@[reassoc (attr := simp)]
lemma germ_fromStalk {𝓕 : Presheaf C X} {c : C} (f : 𝓕 ⟶ skyscraperPresheaf p₀ c) (U) (hU) :
    𝓕.germ U p₀ hU ≫ fromStalk p₀ f = f.app (op U) ≫ eqToHom (if_pos hU) :=
  colimit.ι_desc _ _


theorem to_skyscraper_fromStalk {𝓕 : Presheaf C X} {c : C} (f : 𝓕 ⟶ skyscraperPresheaf p₀ c) :
    toSkyscraperPresheaf p₀ (fromStalk _ f) = f := by
  /-
    X : TopCat
    p₀ : ↑X
    inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝² : CategoryTheory.Category.{u, v} C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasColimits C
    𝓕 : TopCat.Presheaf C X
    c : C
    f : Quiver.Hom 𝓕 (skyscraperPresheaf p₀ c)
    ⊢ Eq (StalkSkyscraperPresheafAdjunctionAuxs.toSkyscraperPresheaf p₀ (StalkSkys …
  -/
  apply NatTrans.ext
  /-
    case app
    X : TopCat
    p₀ : ↑X
    inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝² : CategoryTheory.Category.{u, v} C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasColimits C
    𝓕 : TopCat.Presheaf C X
    c : C
    f : Quiver.Hom 𝓕 (skyscraperPresheaf p₀ c)
    ⊢ Eq (StalkSkyscraperPresheafAdjunctionAuxs.toSkyscraperPresheaf p₀ (StalkSkys …
  -/
  ext U
  /-
    case app.h
    X : TopCat
    p₀ : ↑X
    inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝² : CategoryTheory.Category.{u, v} C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasColimits C
    𝓕 : TopCat.Presheaf C X
    c : C
    f : Quiver.Hom 𝓕 (skyscraperPresheaf p₀ c)
    U : Opposite (TopologicalSpace.Opens ↑X)
    ⊢ Eq ((StalkSkyscraperPresheafAdjunctionAuxs.toSkyscraperPresheaf p₀ (StalkSky …
  -/
  dsimp
  /-
    case app.h
    X : TopCat
    p₀ : ↑X
    inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝² : CategoryTheory.Category.{u, v} C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasColimits C
    𝓕 : TopCat.Presheaf C X
    c : C
    f : Quiver.Hom 𝓕 (skyscraperPresheaf p₀ c)
    U : Opposite (TopologicalSpace.Opens ↑X)
    ⊢ Eq (dite (Membership.mem (Opposite.unop U) p₀) (fun h => CategoryTheory.Cate …
  -/
  split_ifs with h
  · rw [← Category.assoc, germ_fromStalk, Category.assoc, eqToHom_trans, eqToHom_refl,
      Category.comp_id]
    /-
      case neg
      X : TopCat
      p₀ : ↑X
      inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      𝓕 : TopCat.Presheaf C X
      c : C
      f : Quiver.Hom 𝓕 (skyscraperPresheaf p₀ c)
      U : Opposite (TopologicalSpace.Opens ↑X)
      h : Not (Membership.mem (Opposite.unop U) p₀)
      ⊢ Eq ((Eq.rec CategoryTheory.Limits.terminalIsTerminal ⋯).from (𝓕.obj U)) (f.a …
    -/
  · exact ((if_neg h).symm.ndrec terminalIsTerminal).hom_ext ..
    /-
      🎉 no goals
    -/


theorem fromStalk_to_skyscraper {𝓕 : Presheaf C X} {c : C} (f : 𝓕.stalk p₀ ⟶ c) :
    fromStalk p₀ (toSkyscraperPresheaf _ f) = f := by
  /-
    X : TopCat
    p₀ : ↑X
    inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
    C : Type v
    inst✝² : CategoryTheory.Category.{u, v} C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasColimits C
    𝓕 : TopCat.Presheaf C X
    c : C
    f : Quiver.Hom (𝓕.stalk p₀) c
    ⊢ Eq (StalkSkyscraperPresheafAdjunctionAuxs.fromStalk p₀ (StalkSkyscraperPresh …
  -/
  refine 𝓕.stalk_hom_ext fun U hxU ↦ ?_
  rw [germ_fromStalk, toSkyscraperPresheaf_app, dif_pos hxU, Category.assoc, Category.assoc,
    eqToHom_trans, eqToHom_refl, Category.comp_id, Presheaf.germ]


/-- The unit in `Presheaf.stalkFunctor ⊣ skyscraperPresheafFunctor`
-/
@[simps]
protected def unit :
    𝟭 (Presheaf C X) ⟶ Presheaf.stalkFunctor C p₀ ⋙ skyscraperPresheafFunctor p₀ where
  app _ := toSkyscraperPresheaf _ <| 𝟙 _
  naturality 𝓕 𝓖 f := by
    /-
      X : TopCat
      p₀ : ↑X
      inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      𝓕 𝓖 : TopCat.Presheaf C X
      f : Quiver.Hom 𝓕 𝓖
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (TopCat.P …
    -/
    ext U; dsimp
    /-
      case w
      X : TopCat
      p₀ : ↑X
      inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      𝓕 𝓖 : TopCat.Presheaf C X
      f : Quiver.Hom 𝓕 𝓖
      U : TopologicalSpace.Opens ↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := U }) (dite (Membersh …
    -/
    split_ifs with h
    · simp only [Category.id_comp, Category.assoc, eqToHom_trans_assoc, eqToHom_refl,
        Presheaf.stalkFunctor_map_germ_assoc, Presheaf.stalkFunctor_obj]
      /-
        case neg
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        𝓕 𝓖 : TopCat.Presheaf C X
        f : Quiver.Hom 𝓕 𝓖
        U : TopologicalSpace.Opens ↑X
        h : Not (Membership.mem U p₀)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := U }) ((Eq.rec Catego …
      -/
    · apply ((if_neg h).symm.ndrec terminalIsTerminal).hom_ext
      /-
        🎉 no goals
      -/


/-- The counit in `Presheaf.stalkFunctor ⊣ skyscraperPresheafFunctor`
-/
@[simps]
protected def counit :
    skyscraperPresheafFunctor p₀ ⋙ (Presheaf.stalkFunctor C p₀ : Presheaf C X ⥤ C) ⟶ 𝟭 C where
  app c := (skyscraperPresheafStalkOfSpecializes p₀ c specializes_rfl).hom
                                                                     /-
                                                                       X : TopCat
                                                                       p₀ : ↑X
                                                                       inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
                                                                       C : Type v
                                                                       inst✝² : CategoryTheory.Category.{u, v} C
                                                                       A : C
                                                                       inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                                                       inst✝ : CategoryTheory.Limits.HasColimits C
                                                                       x y : C
                                                                       f : Quiver.Hom x y
                                                                       U : TopologicalSpace.Opens ↑X
                                                                       hxU : Membership.mem U p₀
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (((skyscraperPresheafFunctor p₀).obj  …
                                                                     -/
  naturality x y f := TopCat.Presheaf.stalk_hom_ext _ fun U hxU ↦ by simp [hxU]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- `skyscraperPresheafFunctor` is the right adjoint of `Presheaf.stalkFunctor`
-/
def skyscraperPresheafStalkAdjunction [HasColimits C] :
    (Presheaf.stalkFunctor C p₀ : Presheaf C X ⥤ C) ⊣ skyscraperPresheafFunctor p₀ where
  unit := StalkSkyscraperPresheafAdjunctionAuxs.unit _
  counit := StalkSkyscraperPresheafAdjunctionAuxs.counit _
  left_triangle_components X := by
    /-
      X✝ : TopCat
      p₀ : ↑X✝
      inst✝³ : (U : TopologicalSpace.Opens ↑X✝) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X : TopCat.Presheaf C X✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((TopCat.Presheaf.stalkFunctor C p₀). …
    -/
    dsimp [Presheaf.stalkFunctor, toSkyscraperPresheaf]
    /-
      X✝ : TopCat
      p₀ : ↑X✝
      inst✝³ : (U : TopologicalSpace.Opens ↑X✝) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X : TopCat.Presheaf C X✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimMap (Cate …
    -/
    ext
    simp only [Functor.comp_obj, Functor.op_obj, ι_colimMap_assoc, skyscraperPresheaf_obj,
      whiskerLeft_app, Category.comp_id]
    /-
      case w
      X✝ : TopCat
      p₀ : ↑X✝
      inst✝³ : (U : TopologicalSpace.Opens ↑X✝) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      X : TopCat.Presheaf C X✝
      j✝ : Opposite (TopologicalSpace.OpenNhds p₀)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Membership.mem ((TopologicalSp …
    -/
    split_ifs with h
      /-
        case pos
        X✝ : TopCat
        p₀ : ↑X✝
        inst✝³ : (U : TopologicalSpace.Opens ↑X✝) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        X : TopCat.Presheaf C X✝
        j✝ : Opposite (TopologicalSpace.OpenNhds p₀)
        h : Membership.mem ((TopologicalSpace.OpenNhds.inclusion p₀).obj (Opposite.uno …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp [skyscraperPresheafStalkOfSpecializes]
      /-
        case pos
        X✝ : TopCat
        p₀ : ↑X✝
        inst✝³ : (U : TopologicalSpace.Opens ↑X✝) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        X : TopCat.Presheaf C X✝
        j✝ : Opposite (TopologicalSpace.OpenNhds p₀)
        h : Membership.mem ((TopologicalSpace.OpenNhds.inclusion p₀).obj (Opposite.uno …
        ⊢ Eq (X.germ ((TopologicalSpace.OpenNhds.inclusion p₀).obj (Opposite.unop j✝)) …
      -/
      rfl
      /-
        🎉 no goals
      -/
    · simp only [skyscraperPresheafStalkOfSpecializes, colimit.isoColimitCocone_ι_hom,
        skyscraperPresheafCoconeOfSpecializes_pt, skyscraperPresheafCoconeOfSpecializes_ι_app,
        Functor.comp_obj, Functor.op_obj, skyscraperPresheaf_obj, Functor.const_obj_obj]
      /-
        case neg
        X✝ : TopCat
        p₀ : ↑X✝
        inst✝³ : (U : TopologicalSpace.Opens ↑X✝) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        X : TopCat.Presheaf C X✝
        j✝ : Opposite (TopologicalSpace.OpenNhds p₀)
        h : Not (Membership.mem ((TopologicalSpace.OpenNhds.inclusion p₀).obj (Opposit …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Eq.rec CategoryTheory.Limits.termin …
      -/
      rw [comp_eqToHom_iff]
      /-
        case neg
        X✝ : TopCat
        p₀ : ↑X✝
        inst✝³ : (U : TopologicalSpace.Opens ↑X✝) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        X : TopCat.Presheaf C X✝
        j✝ : Opposite (TopologicalSpace.OpenNhds p₀)
        h : Not (Membership.mem ((TopologicalSpace.OpenNhds.inclusion p₀).obj (Opposit …
        ⊢ Eq ((Eq.rec CategoryTheory.Limits.terminalIsTerminal ⋯).from (X.obj { unop : …
      -/
      apply ((if_neg h).symm.ndrec terminalIsTerminal).hom_ext
      /-
        🎉 no goals
      -/
  right_triangle_components Y := by
    /-
      X : TopCat
      p₀ : ↑X
      inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((StalkSkyscraperPresheafAdjunctionAu …
    -/
    ext
    simp only [skyscraperPresheafFunctor_obj, Functor.id_obj, skyscraperPresheaf_obj,
      Functor.comp_obj, Presheaf.stalkFunctor_obj, unit_app, counit_app,
      skyscraperPresheafStalkOfSpecializes, skyscraperPresheafFunctor_map, Presheaf.comp_app,
      toSkyscraperPresheaf_app, Category.id_comp, SkyscraperPresheafFunctor.map'_app]
    /-
      case w
      X : TopCat
      p₀ : ↑X
      inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
      C : Type v
      inst✝² : CategoryTheory.Category.{u, v} C
      A : C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasColimits C
      Y : C
      U✝ : TopologicalSpace.Opens ↑X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Membership.mem U✝ p₀) (fun h = …
    -/
    split_ifs with h
      /-
        case pos
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        Y : C
        U✝ : TopologicalSpace.Opens ↑X
        h : Membership.mem U✝ p₀
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp [Presheaf.germ]
      /-
        case pos
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        Y : C
        U✝ : TopologicalSpace.Opens ↑X
        h : Membership.mem U✝ p₀
        ⊢ Eq (CategoryTheory.CategoryStruct.id (ite (Membership.mem U✝ p₀) Y (Category …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        Y : C
        U✝ : TopologicalSpace.Opens ↑X
        h : Not (Membership.mem U✝ p₀)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Eq.rec CategoryTheory.Limits.termin …
      -/
    · simp
      /-
        case neg
        X : TopCat
        p₀ : ↑X
        inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
        C : Type v
        inst✝² : CategoryTheory.Category.{u, v} C
        A : C
        inst✝¹ : CategoryTheory.Limits.HasTerminal C
        inst✝ : CategoryTheory.Limits.HasColimits C
        Y : C
        U✝ : TopologicalSpace.Opens ↑X
        h : Not (Membership.mem U✝ p₀)
        ⊢ Eq (CategoryTheory.CategoryStruct.id (ite (Membership.mem U✝ p₀) Y (Category …
      -/
      rfl
      /-
        🎉 no goals
      -/


instance [HasColimits C] : (skyscraperPresheafFunctor p₀ : C ⥤ Presheaf C X).IsRightAdjoint  :=
  (skyscraperPresheafStalkAdjunction _).isRightAdjoint


instance [HasColimits C] : (Presheaf.stalkFunctor C p₀).IsLeftAdjoint  :=
  -- Use a classical instance instead of the one from `variable`s
  have : ∀ U : Opens X, Decidable (p₀ ∈ U) := fun _ ↦ Classical.dec _
  (skyscraperPresheafStalkAdjunction _).isLeftAdjoint


/-- Taking stalks of a sheaf is the left adjoint functor to `skyscraperSheafFunctor`
-/
def stalkSkyscraperSheafAdjunction [HasColimits C] :
    Sheaf.forget C X ⋙ Presheaf.stalkFunctor _ p₀ ⊣ skyscraperSheafFunctor p₀ where
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext1` is changed to `Sheaf.Hom.ext`,
  unit :=
    { app := fun 𝓕 => ⟨(StalkSkyscraperPresheafAdjunctionAuxs.unit p₀).app 𝓕.1⟩
      naturality := fun 𝓐 𝓑 f => Sheaf.Hom.ext <| by
        /-
          X : TopCat
          p₀ : ↑X
          inst✝³ : (U : TopologicalSpace.Opens ↑X) → Decidable (Membership.mem U p₀)
          C : Type v
          inst✝² : CategoryTheory.Category.{u, v} C
          A : C
          inst✝¹ : CategoryTheory.Limits.HasTerminal C
          inst✝ : CategoryTheory.Limits.HasColimits C
          𝓐 𝓑 : TopCat.Sheaf C X
          f : Quiver.Hom 𝓐 𝓑
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (TopCat.S …
        -/
        apply (StalkSkyscraperPresheafAdjunctionAuxs.unit p₀).naturality }
        /-
          🎉 no goals
        -/
  counit := StalkSkyscraperPresheafAdjunctionAuxs.counit p₀
  left_triangle_components X :=
    ((skyscraperPresheafStalkAdjunction p₀).left_triangle_components X.val)
  right_triangle_components _ :=
    Sheaf.Hom.ext ((skyscraperPresheafStalkAdjunction p₀).right_triangle_components _)


instance [HasColimits C] : (skyscraperSheafFunctor p₀ : C ⥤ Sheaf C X).IsRightAdjoint  :=
  (stalkSkyscraperSheafAdjunction _).isRightAdjoint


